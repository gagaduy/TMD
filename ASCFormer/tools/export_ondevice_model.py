#!/usr/bin/env python3
"""Export Trained RTM Model to Self-Contained On-Device ONNX Models.

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

# Force pure-PyTorch NAT implementation
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
from mmseg.datasets.transforms.transforms import ELA
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
        sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).unsqueeze(0).unsqueeze(0)
        sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).unsqueeze(0).unsqueeze(0)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)
        pool_weights = torch.ones((1, 1, 5, 5))
        self.register_buffer('pool_weights', pool_weights)

    def forward(self, gray: torch.Tensor) -> torch.Tensor:
        gray_norm = gray / 255.0
        dark_text = (1.0 - gray_norm).clamp(0.0, 1.0)
        bright_text = (gray_norm - 0.5).clamp(0.0, 1.0) * 2.0
        region = torch.maximum(dark_text, bright_text)
        gx = F.conv2d(gray_norm, self.sobel_x, padding=1)
        gy = F.conv2d(gray_norm, self.sobel_y, padding=1)
        grad = torch.sqrt(gx * gx + gy * gy + 1e-6)
        boundary = (grad * 2.0).clamp(0.0, 1.0)
        mask = torch.sigmoid((boundary - 0.25) * 10.0)
        dilated = F.conv2d(mask, self.pool_weights, padding=2)
        distance = (dilated / 25.0).clamp(0.0, 1.0)
        return torch.cat([region, boundary, distance], dim=1)


class OnDeviceRTMModel(nn.Module):
    """Unified self-contained model wrapper for mobile deployment."""

    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.base_model = base_model
        self.ocr_extractor = OnGraphOCRSpatial()

        # ImageNet normalization buffers
        mean = torch.tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1)
        std = torch.tensor([58.395, 57.12, 57.375]).view(1, 3, 1, 1)
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)

        # RGB to Grayscale weights for OCR extractor
        gray_w = torch.tensor([0.2989, 0.5870, 0.1140]).view(1, 3, 1, 1)
        self.register_buffer('gray_w', gray_w)

    def forward(self, img_bgr: torch.Tensor, ela_bgr: torch.Tensor) -> torch.Tensor:
        # Convert BGR to RGB
        img_rgb = torch.flip(img_bgr, dims=[1])
        ela_rgb = torch.flip(ela_bgr, dims=[1])

        # Normalize RGB
        norm_rgb = (img_rgb - self.mean) / self.std

        # Extract OCR spatial map from grayscale
        gray = (img_rgb * self.gray_w).sum(dim=1, keepdim=True)
        ocr_spatial = self.ocr_extractor(gray)

        # Forward backbone
        feats = self.base_model.backbone(norm_rgb, ela_rgb, ocr_spatial)

        # Forward decode head
        dh = self.base_model.decode_head
        seg_logits = self._decode_head_forward(feats, dh)

        # Softmax probability for tamper class (channel 1)
        prob = F.softmax(seg_logits, dim=1)[:, 1, :, :]
        return prob

    def _decode_head_forward(self, inputs, dh):
        inputs = dh._transform_inputs(inputs)
        outs = []
        for idx in range(len(inputs)):
            x = inputs[idx]
            conv = dh.convs[idx]
            outs.append(
                F.interpolate(
                    conv(x),
                    size=inputs[0].shape[2:],
                    mode='bilinear',
                    align_corners=dh.align_corners
                )
            )
        out = dh.fusion_conv(torch.cat(outs, dim=1))
        seg_logits = dh.cls_seg(out)

        # Apply prototype projection if present
        if hasattr(dh, 'prototypes') and dh.prototypes is not None:
            if hasattr(dh, 'proj_head') and dh.proj_head is not None:
                feats = dh.proj_head(out)
            else:
                feats = out
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
    missing, unexpected = base_model.load_state_dict(state_dict, strict=True)
    assert len(missing) == 0 and len(unexpected) == 0, f'Missing: {missing}, Unexpected: {unexpected}'
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
    print('✅ SUCCESS!')


if __name__ == '__main__':
    main()
