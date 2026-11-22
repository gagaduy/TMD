# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.registry import MODELS
from ..utils import resize


def _make_ds_conv(in_channels: int,
                  out_channels: int,
                  kernel_size: int = 3,
                  norm_cfg: dict = None,
                  act_cfg: dict = dict(type='ReLU')) -> nn.Sequential:
    """Depthwise-Separable 3x3 Conv block: ~8-10x fewer FLOPs than dense 3x3 Conv."""
    return nn.Sequential(
        ConvModule(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=in_channels,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg),
        ConvModule(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg),
    )


@MODELS.register_module()
class LitePrototypeHead(BaseDecodeHead):
    """Lightweight Dual-Prototype Contrastive Decode Head tailored for MiT-B0 Mobile Full-Scan.

    Eliminates the 83.9% FLOPs bottleneck (194.2 / 231.5 GFLOPs) of the MiT-B2-sized
    PrototypeContrastiveHead by:
      1. Matching decoder width to MiT-B0 capacity (channels=128, dim=64).
      2. Replacing dense 3x3 convolutions at 128x128 & 256x256 with 1x1 channel fusion
         + Depthwise-Separable 3x3 refinement blocks (~35x fewer decoder FLOPs).
      3. Retaining Dual EMA Class Prototypes (P_auth, P_tamp) with Prototypical Logit
         Fusion and native support for compound losses (CrossEntropy + Dice).
    """

    def __init__(self,
                 interpolate_mode='bilinear',
                 dim=64,
                 temperature=0.1,
                 momentum=0.99,
                 loss_weight_contrastive=0.05,
                 sep_margin=0.0,
                 loss_weight_sep=0.01,
                 proto_fuse_weight=0.5,
                 up_decode=True,
                 **kwargs):
        legacy_keys = ['use_memory', 'max_memory_step', 'cl_sampler', 'max_points',
                       'max_memory_size', 'loss_const', 'min_points', 'batch_cl',
                       'multi_layer_cl', 'save_feat', 'upsample_first', 'field_mode']
        for k in legacy_keys:
            kwargs.pop(k, None)

        super().__init__(input_transform='multiple_select', **kwargs)

        self.interpolate_mode = interpolate_mode
        self.up_decode = up_decode
        self.dim = dim
        self.temperature = temperature
        self.momentum = momentum
        self.loss_weight_contrastive = loss_weight_contrastive
        self.sep_margin = sep_margin
        self.loss_weight_sep = loss_weight_sep
        self.proto_fuse_weight = proto_fuse_weight

        num_inputs = len(self.in_channels)
        assert num_inputs == len(self.in_index)

        # Multi-scale 1x1 lateral projections
        self.convs = nn.ModuleList()
        for i in range(num_inputs):
            self.convs.append(
                ConvModule(
                    in_channels=self.in_channels[i],
                    out_channels=self.channels,
                    kernel_size=1,
                    stride=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))

        # Lightweight 1x1 channel fusion + Depthwise-Separable 3x3 spatial refinement
        self.fusion_conv = nn.Sequential(
            ConvModule(
                in_channels=self.channels * num_inputs,
                out_channels=self.channels,
                kernel_size=1,
                stride=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg),
            _make_ds_conv(
                in_channels=self.channels,
                out_channels=self.channels,
                kernel_size=3,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg),
        )

        # Lightweight contrastive projection head: channels -> DW3x3 -> dim
        self.cc_proj = nn.Sequential(
            _make_ds_conv(
                in_channels=self.channels,
                out_channels=self.channels,
                kernel_size=3,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg),
            ConvModule(
                in_channels=self.channels,
                out_channels=dim,
                kernel_size=1,
                norm_cfg=None,
                act_cfg=None),
        )

        # Lightweight high-resolution segmentation projection at 256x256
        self.seg_proj = _make_ds_conv(
            in_channels=self.channels,
            out_channels=self.channels,
            kernel_size=3,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        # Dual EMA Class Prototypes (0=authentic, 1=tampered)
        self.register_buffer('prototypes', torch.zeros(2, dim))
        self.register_buffer('proto_initialized', torch.tensor([False, False]))

        print(f'>>>> LitePrototypeHead initialized (channels={self.channels}, dim={dim}, DW-Sep 3x3) <<<<')

    def forward(self, inputs):
        inputs = self._transform_inputs(inputs)
        outs = []
        target_size = inputs[0].shape[2:]
        for idx in range(len(inputs)):
            conv_x = self.convs[idx](inputs[idx])
            if idx > 0:
                conv_x = resize(
                    input=conv_x,
                    size=target_size,
                    mode=self.interpolate_mode,
                    align_corners=self.align_corners)
            outs.append(conv_x)

        out = self.fusion_conv(torch.cat(outs, dim=1))
        feats = self.cc_proj(out)

        seg_out = out
        if self.up_decode:
            seg_out = resize(
                input=seg_out,
                size=[s * 2 for s in seg_out.shape[2:]],
                mode=self.interpolate_mode,
                align_corners=self.align_corners)
        seg_out = self.seg_proj(seg_out)
        seg_logits = self.cls_seg(seg_out)

        if self.proto_initialized.all():
            feats_norm = F.normalize(feats, dim=1)
            prototypes = F.normalize(self.prototypes, dim=1)
            proto_logits = torch.einsum('bchw, kc -> bkhw', feats_norm, prototypes) / self.temperature
            proto_logits = resize(
                proto_logits,
                size=seg_logits.shape[2:],
                mode=self.interpolate_mode,
                align_corners=self.align_corners)
            seg_logits = seg_logits + self.proto_fuse_weight * proto_logits

        return seg_logits, feats

    def forward_infer(self, inputs):
        return self.forward(inputs)

    def predict(self, inputs, batch_img_metas, test_cfg):
        seg_logits, _ = self.forward(inputs)
        return self.predict_by_feat(seg_logits, batch_img_metas)

    def loss_from_outputs(self, seg_logits, feats, batch_data_samples) -> dict:
        losses = self.loss_by_feat(seg_logits, batch_data_samples)
        proto_losses = self.loss_by_prototype(feats, batch_data_samples)
        losses.update(proto_losses)
        return losses

    def loss(self, inputs, batch_data_samples, train_cfg) -> dict:
        seg_logits, feats = self.forward(inputs)
        return self.loss_from_outputs(seg_logits, feats, batch_data_samples)

    def loss_by_prototype(self, feats, batch_data_samples):
        B, C, H, W = feats.size()
        seg_label = self._stack_batch_gt(batch_data_samples)
        seg_label = F.interpolate(seg_label.float(), size=(H, W), mode='nearest').long()

        feats_norm = F.normalize(feats.float(), dim=1)
        feats_flat = feats_norm.permute(0, 2, 3, 1).contiguous().view(-1, C)
        labels_flat = seg_label.view(-1)

        with torch.no_grad():
            for c in range(2):
                mask_c = (labels_flat == c)
                if mask_c.sum() > 0:
                    center_c = feats_flat[mask_c].mean(dim=0)
                    center_c = F.normalize(center_c, dim=0)
                    if not self.proto_initialized[c]:
                        self.prototypes[c] = center_c
                        self.proto_initialized[c] = True
                    else:
                        updated = self.momentum * self.prototypes[c] + (1.0 - self.momentum) * center_c
                        self.prototypes[c] = F.normalize(updated, dim=0)

        if not self.proto_initialized.all():
            return {'loss_contrastive': feats.new_tensor(0.0)}

        prototypes = F.normalize(self.prototypes.float(), dim=1)
        proto_logits = torch.einsum('bchw, kc -> bkhw', feats_norm, prototypes) / self.temperature
        loss_proto = F.cross_entropy(proto_logits, seg_label.squeeze(1), ignore_index=self.ignore_index)

        cos_sim = torch.dot(prototypes[0], prototypes[1])
        loss_sep = torch.clamp(cos_sim - self.sep_margin, min=0.0)

        total_loss = self.loss_weight_contrastive * loss_proto + self.loss_weight_sep * loss_sep
        return {'loss_contrastive': total_loss}
