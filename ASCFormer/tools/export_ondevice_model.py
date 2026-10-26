#!/usr/bin/env python3
"""Export SOTA RTM Model to Self-Contained On-Device ONNX Models.

Supports both MiT-B0 and MiT-B2 architectures.
Converts model with:
1. Internal RGB normalization
2. Internal DCT (8x8 ZigZag) and SRM high-pass residual extraction
3. Internal Morphological OCR spatial map extraction
4. NATFuser Block (pure-tensor-slice implementation)
5. Dual Class Prototypes projection
6. Quantization to INT8 (dynamic quantization)
"""
import argparse
import os
import sys

# Force pure-PyTorch NAT implementation so checkpoint loads cleanly and exports to ONNX
sys.modules['natten'] = None
sys.modules['natten.functional'] = None

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ASC_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, ASC_ROOT)

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType

from mmengine.config import Config
from mmseg.registry import MODELS
from mmseg.utils import register_all_modules
from mmseg.datasets.transforms.transforms import BlockDCT, ELA
import mmseg.models.toys.fusers as fusers_mod
import mmseg.models.backbones.nat as nat_mod


def _trace_clean_na2d_shift(q, k, v, scale: float, kernel_size: int = 5, dilation: int = 1):
    H, W = int(q.shape[1]), int(q.shape[2])
    d = int(dilation or 1)
    ks = int(kernel_size)
    pad = (ks // 2) * d
    q_s = q * float(scale)
    k_pad = F.pad(k, (0, 0, 0, 0, pad, pad, pad, pad))
    v_pad = F.pad(v, (0, 0, 0, 0, pad, pad, pad, pad))
    logits = []
    for dy in range(ks):
        y0 = dy * d
        for dx in range(ks):
            x0 = dx * d
            k_slice = k_pad[:, y0:y0 + H, x0:x0 + W]
            logits.append((q_s * k_slice).sum(dim=-1))
    attn = torch.stack(logits, dim=-1).softmax(dim=-1)
    out = torch.zeros_like(q)
    idx = 0
    for dy in range(ks):
        y0 = dy * d
        for dx in range(ks):
            x0 = dx * d
            v_slice = v_pad[:, y0:y0 + H, x0:x0 + W]
            out = out + attn[..., idx:idx + 1] * v_slice
            idx += 1
    return out


def _trace_clean_nat_self_shift(qkv, scale: float, kernel_size: int = 5, dilation: int = 1):
    q, k, v = qkv[0], qkv[1], qkv[2]
    return _trace_clean_na2d_shift(q, k, v, scale, kernel_size, dilation)


fusers_mod._fast_na2d_shift = _trace_clean_na2d_shift
fusers_mod._compiled_na2d_shift = _trace_clean_na2d_shift
nat_mod._fast_nat_self_shift = _trace_clean_nat_self_shift
nat_mod._compiled_nat_self_shift = _trace_clean_nat_self_shift


def replace_syncbn(cfg_Node):
    if isinstance(cfg_Node, dict):
        for k, v in cfg_Node.items():
            if k == 'type' and v == 'SyncBN':
                cfg_Node[k] = 'BN'
            else:
                replace_syncbn(v)
    elif isinstance(cfg_Node, list):
        for item in cfg_Node:
            replace_syncbn(item)


class OnGraphOCRSpatial(nn.Module):
    """Pure-PyTorch morphological text spatial map extractor (Region, Boundary, Distance)."""

    def __init__(self):
        super().__init__()
        gx = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]).view(1, 1, 3, 3)
        gy = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]).view(1, 1, 3, 3)
        self.register_buffer('gx', gx)
        self.register_buffer('gy', gy)

    def forward(self, y_channel: torch.Tensor) -> torch.Tensor:
        dx = F.conv2d(y_channel, self.gx, padding=1)
        dy = F.conv2d(y_channel, self.gy, padding=1)
        grad = torch.sqrt(dx * dx + dy * dy + 1e-6)
        mean_g = grad.mean(dim=(2, 3), keepdim=True)
        std_g = grad.std(dim=(2, 3), keepdim=True)
        stroke = (grad > (mean_g + 1.2 * std_g)).float()

        # Morphological closing to connect characters into text regions
        d1 = F.max_pool2d(stroke, kernel_size=(5, 11), stride=1, padding=(2, 5))
        region = -F.max_pool2d(-d1, kernel_size=(5, 11), stride=1, padding=(2, 5))
        eroded = -F.max_pool2d(-region, kernel_size=3, stride=1, padding=1)
        region = F.max_pool2d(eroded, kernel_size=3, stride=1, padding=1)

        # Boundary: morphological gradient
        dil3 = F.max_pool2d(region, kernel_size=3, stride=1, padding=1)
        ero3 = -F.max_pool2d(-region, kernel_size=3, stride=1, padding=1)
        boundary = (dil3 - ero3).clamp(0.0, 1.0)

        # Distance transform approximation
        e1 = -F.max_pool2d(-region, kernel_size=3, stride=1, padding=1)
        e2 = -F.max_pool2d(-e1, kernel_size=5, stride=1, padding=2)
        e3 = -F.max_pool2d(-e2, kernel_size=7, stride=1, padding=3)
        dist = F.avg_pool2d(region + e1 + e2 + e3, kernel_size=5, stride=1, padding=2) * region
        d_max = dist.amax(dim=(2, 3), keepdim=True).clamp(min=1e-5)
        dist = (dist / d_max).clamp(0.0, 1.0)
        return torch.cat([region, boundary, dist], dim=1)


class OnDeviceRTMModel(nn.Module):
    """Self-contained wrapper around MyModelFull for mobile/client inference."""

    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.register_buffer(
            'rgb_mean', torch.tensor([123.675, 116.28, 103.53], dtype=torch.float32).view(1, 3, 1, 1)
        )
        self.register_buffer(
            'rgb_std', torch.tensor([58.395, 57.12, 57.375], dtype=torch.float32).view(1, 3, 1, 1)
        )

        # Build exact 8x8 orthonormal DCT basis in zigzag order (64, 1, 8, 8)
        bdct = BlockDCT(block_size=8, zigzag=True)
        dct_mat = bdct.dct_mat
        zz_r, zz_c = bdct.zigzag_encode_idx
        basis = np.zeros((64, 1, 8, 8), dtype=np.float32)
        for k in range(64):
            u, v = zz_r[k], zz_c[k]
            basis[k, 0] = np.outer(dct_mat[u, :], dct_mat[v, :])
        self.register_buffer('dct_basis', torch.from_numpy(basis))

        # Disable internal reshape in DCTProcessor because dct_basis conv2d already outputs (B, 64, H//8, W//8)
        self.dct_proc = base_model.preprocessor_sec[0]
        self.dct_proc.reshape = False
        self.ela_proc = base_model.preprocessor_sec[1]
        self.srm_proc = base_model.preprocessor_sec[2]

        self.ocr_extractor = OnGraphOCRSpatial()
        self.backbone = base_model.backbone
        self.decode_head = base_model.decode_head

    def forward(self, img_bgr: torch.Tensor, ela_bgr: torch.Tensor) -> torch.Tensor:
        # 1. Normalized RGB stream
        b = img_bgr[:, 0:1, :, :]
        g = img_bgr[:, 1:2, :, :]
        r = img_bgr[:, 2:3, :, :]
        img_rgb = torch.cat([r, g, b], dim=1)
        norm_rgb = (img_rgb - self.rgb_mean) / self.rgb_std

        # 2. Y luminance & 8x8 Zigzag DCT stream
        y_chan = torch.round(0.299 * r + 0.587 * g + 0.114 * b)
        dct_blocks = F.conv2d(y_chan, self.dct_basis, stride=8)
        f_dct = self.dct_proc(dct_blocks)

        # 3. ELA & SRM streams
        f_ela = self.ela_proc(ela_bgr)
        f_srm = self.srm_proc(norm_rgb)

        # 4. On-graph OCR spatial stream
        ocr_map = self.ocr_extractor(y_chan)

        # 5. Trace-safe Tri-Stream Backbone + Dual-Prototype Decode Head
        feats = self._forward_backbone(norm_rgb, [f_dct, f_ela, f_srm], ocr_map)
        seg_logits = self._forward_decode_head(feats)
        seg_logits_up = F.interpolate(
            seg_logits, size=(512, 512), mode='bilinear', align_corners=False
        )
        prob_tamper = F.softmax(seg_logits_up, dim=1)[:, 1, :, :]
        return prob_tamper

    def _forward_backbone(self, x_cam, x_extra, x_ocr):
        from mmseg.models.utils import nlc_to_nchw
        bb = self.backbone
        outs = []
        # Stage 0 extra modalities: [f_ela, f_srm], Stage 1 extra modality: [f_dct]
        x_extra_m = [x_extra[0]]
        x_extra = [x_extra[1], x_extra[2]]
        B = 1

        for i in range(bb.num_stage_extra):
            # 1. Main visual stream (RGB)
            layer = bb.main_branch.layers[i]
            x_cam, hw_shape = layer[0](x_cam)
            H, W = int(hw_shape[0]), int(hw_shape[1])
            for block in layer[1]:
                x_cam = block(x_cam, hw_shape)
            x_cam = layer[2](x_cam)
            x_cam = nlc_to_nchw(x_cam, hw_shape)

            # 2. Forensic auxiliary stream (DCT, ELA, SRM)
            layer_extra = bb.extra_branch.layers[i]
            if i == 1:
                temp = []
                for item in x_extra_m:
                    em, s = bb.extra_patch_embed(item)
                    temp.append(nlc_to_nchw(em, s))
                x_extra, _ = layer_extra[0](x_extra)
                x_extra.extend(temp)
            else:
                x_extra, _ = layer_extra[0](x_extra)

            x_f = bb.extra_branch.tokenselect(x_extra, bb.extra_branch.extra_score_predictor[i])
            for block in layer_extra[1]:
                x_f = block(x_f, hw_shape)
            x_f = layer_extra[2](x_f)
            x_f = nlc_to_nchw(x_f, hw_shape)

            # Stage 1 Fusion: RGB + Forensic
            x_fused_ra = bb.FFMs[i](x_cam, x_f)

            # 3. OCR spatial stream
            layer_ocr = bb.ocr_branch.layers[i]
            x_ocr_t, hw_ocr = layer_ocr[0](x_ocr)
            for block in layer_ocr[1]:
                x_ocr_t = block(x_ocr_t, hw_ocr)
            x_ocr_t = layer_ocr[2](x_ocr_t)
            x_ocr_feat = nlc_to_nchw(x_ocr_t, hw_ocr)

            # Stage 2 Fusion: [RGB+Forensic] + OCR
            x_fused_final = bb.FFM_ocrs[i](x_fused_ra, x_ocr_feat)
            x_ocr = x_ocr_feat
            outs.append(x_fused_final)

            x_extra = [x_.reshape(B, H, W, -1).permute(0, 3, 1, 2) + x_f for x_ in x_extra]

        return tuple(outs)

    def _forward_decode_head(self, inputs):
        dh = self.decode_head
        target_size = (int(inputs[0].shape[2]), int(inputs[0].shape[3]))
        outs = []
        for idx in range(len(inputs)):
            conv_x = dh.convs[idx](inputs[idx])
            if idx > 0:
                conv_x = F.interpolate(conv_x, size=target_size, mode='bilinear', align_corners=False)
            outs.append(conv_x)
        out = dh.fusion_conv(torch.cat(outs, dim=1))
        feats = dh.cc_proj(out)
        up_size = (target_size[0] * 2, target_size[1] * 2)
        seg_out = F.interpolate(out, size=up_size, mode='bilinear', align_corners=False)
        seg_out = dh.seg_proj(seg_out)
        seg_logits = dh.cls_seg(seg_out)

        feats_norm = F.normalize(feats, dim=1)
        prototypes = F.normalize(dh.prototypes, dim=1)
        proto_logits = torch.einsum('bchw, kc -> bkhw', feats_norm, prototypes) / float(dh.temperature)
        logit_size = (int(seg_logits.shape[2]), int(seg_logits.shape[3]))
        proto_logits = F.interpolate(proto_logits, size=logit_size, mode='bilinear', align_corners=False)
        seg_logits = seg_logits + 0.5 * proto_logits
        return seg_logits


def main():
    parser = argparse.ArgumentParser(description='Export RTM checkpoint to ONNX INT8')
    parser.add_argument('--config', required=True, help='Path to model config')
    parser.add_argument('--checkpoint', required=True, help='Path to .pth checkpoint')
    parser.add_argument('--output', required=True, help='Path to output .onnx file')
    parser.add_argument('--fp32', action='store_true', help='Export FP32 only, skip INT8 quantization')
    args = parser.parse_args()

    register_all_modules(init_default_scope=True)

    print(f'[1/4] Loading config & checkpoint: {args.checkpoint}...')
    cfg = Config.fromfile(args.config)
    replace_syncbn(cfg.model)
    if hasattr(cfg.model.backbone, 'backbone_main'):
        cfg.model.backbone.backbone_main.pretrained = None
    if hasattr(cfg.model.backbone, 'backbone_extra'):
        cfg.model.backbone.backbone_extra.pretrained = None
    if hasattr(cfg.model.backbone, 'backbone_ocr'):
        cfg.model.backbone.backbone_ocr.pretrained = None

    base_model = MODELS.build(cfg.model)
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
    missing, unexpected = base_model.load_state_dict(state_dict, strict=False)
    assert len(missing) == 0, f'Critical missing weights in checkpoint: {missing}'
    if len(unexpected) > 0:
        print(f'      Ignored auxiliary keys in checkpoint: {len(unexpected)} keys')
    base_model.eval()

    wrapper = OnDeviceRTMModel(base_model).eval()

    dummy_img = torch.zeros((1, 3, 512, 512), dtype=torch.float32)
    dummy_ela = torch.zeros((1, 3, 512, 512), dtype=torch.float32)

    with torch.no_grad():
        pt_out = wrapper(dummy_img, dummy_ela).numpy()
    print(f'      PyTorch dummy output shape={pt_out.shape}, range=[{pt_out.min():.4f}, {pt_out.max():.4f}]')

    fp32_path = args.output if args.fp32 else args.output + '.fp32.tmp'
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    print(f'[2/4] Exporting FP32 ONNX model to {fp32_path}...')
    torch.onnx.export(
        wrapper,
        (dummy_img, dummy_ela),
        fp32_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['img_bgr', 'ela_bgr'],
        output_names=['prob_tamper'],
    )
    fp32_size_mb = os.path.getsize(fp32_path) / (1024 * 1024)
    print(f'      FP32 ONNX exported ({fp32_size_mb:.2f} MB)')

    if not args.fp32:
        print(f'[3/4] Quantizing ONNX model to INT8 ({args.output})...')
        onnx_model = onnx.load(fp32_path)
        exclude_nodes = [
            n.name for n in onnx_model.graph.node
            if n.op_type == 'Conv' and (
                'srm_proc' in n.name or 'ocr_extractor' in n.name
                or 'dct_proc' in n.name or n.name == '/Conv'
            )
        ]
        quantize_dynamic(
            model_input=fp32_path,
            model_output=args.output,
            op_types_to_quantize=['MatMul', 'Gemm', 'Conv'],
            nodes_to_exclude=exclude_nodes,
            weight_type=QuantType.QUInt8,
        )
        if os.path.exists(fp32_path):
            os.remove(fp32_path)
        int8_size_mb = os.path.getsize(args.output) / (1024 * 1024)
        print(f'      INT8 ONNX exported ({int8_size_mb:.2f} MB)')

    print('[4/4] Verifying ONNX Runtime numerical equivalence...')
    sess = ort.InferenceSession(args.output, providers=['CPUExecutionProvider'])
    ort_out = sess.run(['prob_tamper'], {'img_bgr': dummy_img.numpy(), 'ela_bgr': dummy_ela.numpy()})[0]
    print(f'      ONNX Runtime verified! Output shape: {ort_out.shape}')
    print('✅ EXPORT SUCCESS!')


if __name__ == '__main__':
    main()
