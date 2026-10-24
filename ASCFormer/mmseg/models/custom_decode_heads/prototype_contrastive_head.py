# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.registry import MODELS
from ..utils import resize


@MODELS.register_module()
class PrototypeContrastiveHead(BaseDecodeHead):
    """Dual-Prototype Contrastive Decode Head for tampered text detection.

    Replaces the heavy O(N^2) pixel memory bank with dual Class Prototypes:
    - P_auth (c=0): Prototype representing authentic/background regions.
    - P_tamp (c=1): Prototype representing manipulated/tampered text regions.

    Prototypes are updated via Exponential Moving Average (EMA) during training,
    and InfoNCE contrastive loss is computed across all pixel feature vectors.

    Args:
        interpolate_mode (str): Upsample interpolation mode. Default: 'bilinear'.
        dim (int): Projection dimension for contrastive features. Default: 128.
        temperature (float): Temperature parameter for cosine similarity scaling. Default: 0.1.
        momentum (float): Momentum coefficient for EMA prototype update. Default: 0.99.
        loss_weight_contrastive (float): Weight for prototype contrastive loss. Default: 0.05.
        sep_margin (float): Separation margin between prototypes. Default: 0.0.
        loss_weight_sep (float): Weight for prototype separation loss. Default: 0.01.
        up_decode (bool): Whether to upsample output before classification. Default: True.
    """

    def __init__(self,
                 interpolate_mode='bilinear',
                 dim=128,
                 temperature=0.1,
                 momentum=0.99,
                 loss_weight_contrastive=0.05,
                 sep_margin=0.0,
                 loss_weight_sep=0.01,
                 up_decode=True,
                 **kwargs):
        # Filter out any legacy memory bank kwargs passed from config inheritance
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

        num_inputs = len(self.in_channels)
        assert num_inputs == len(self.in_index)

        # Multi-scale feature projection convs
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

        # Fusion convolution across all scales
        self.fusion_conv = ConvModule(
            in_channels=self.channels * num_inputs,
            out_channels=self.channels,
            kernel_size=3,
            padding=1,
            norm_cfg=self.norm_cfg)

        # Contrastive projection head: channels -> dim*4 -> dim
        self.cc_proj = nn.Sequential(
            ConvModule(
                in_channels=self.channels,
                out_channels=dim * 4,
                kernel_size=3,
                padding=1,
                norm_cfg=self.norm_cfg),
            ConvModule(
                in_channels=dim * 4,
                out_channels=dim,
                kernel_size=1,
                norm_cfg=None,
                act_cfg=None)
        )

        # Segmentation projection conv
        self.seg_proj = ConvModule(
            in_channels=self.channels,
            out_channels=self.channels,
            kernel_size=3,
            norm_cfg=self.norm_cfg)

        # Register dual class prototypes as persistent buffers (2 classes: 0=auth, 1=tamp)
        self.register_buffer('prototypes', torch.zeros(2, dim))
        self.register_buffer('proto_initialized', torch.tensor([False, False]))

        print('>>>> PrototypeContrastiveHead initialized (Dual EMA Prototypes: Auth & Tamp) <<<<')

    def forward(self, inputs):
        """Forward feature transformation and prediction."""
        inputs = self._transform_inputs(inputs)
        outs = []

        for idx in range(len(inputs)):
            x = inputs[idx]
            conv = self.convs[idx]
            outs.append(
                resize(
                    input=conv(x),
                    size=inputs[0].shape[2:],
                    mode=self.interpolate_mode,
                    align_corners=self.align_corners))

        out = self.fusion_conv(torch.cat(outs, dim=1))

        # Projection for contrastive features: (B, dim, H, W)
        feats = self.cc_proj(out)

        # Segmentation prediction head
        seg_out = out
        if self.up_decode:
            seg_out = resize(
                input=seg_out,
                size=[s * 2 for s in seg_out.shape[2:]],
                mode=self.interpolate_mode,
                align_corners=self.align_corners)
        seg_out = self.seg_proj(seg_out)
        seg_logits = self.cls_seg(seg_out)

        # Prototypical logit fusion: empower segmentation with learned prototypes
        if self.proto_initialized.all():
            feats_norm = F.normalize(feats, dim=1)
            prototypes = F.normalize(self.prototypes, dim=1)
            proto_logits = torch.einsum('bchw, kc -> bkhw', feats_norm, prototypes) / self.temperature
            proto_logits = resize(
                proto_logits,
                size=seg_logits.shape[2:],
                mode=self.interpolate_mode,
                align_corners=self.align_corners)
            seg_logits = seg_logits + 0.5 * proto_logits

        return seg_logits, feats


    def forward_infer(self, inputs):
        """Inference forward pass."""
        return self.forward(inputs)

    def predict(self, inputs, batch_img_metas, test_cfg):
        """Prediction entry point for evaluation."""
        seg_logits, _ = self.forward(inputs)
        return self.predict_by_feat(seg_logits, batch_img_metas)

    def loss(self, inputs, batch_data_samples, train_cfg) -> dict:
        """Forward loss computation during training."""
        seg_logits, feats = self.forward(inputs)

        # 1. Standard segmentation loss (CrossEntropy)
        losses = self.loss_by_feat(seg_logits, batch_data_samples)

        # 2. Prototype-based Contrastive Loss
        proto_losses = self.loss_by_prototype(feats, batch_data_samples)
        losses.update(proto_losses)

        return losses

    def loss_by_prototype(self, feats, batch_data_samples):
        """Compute dual-prototype contrastive InfoNCE loss with EMA prototype updates."""
        B, C, H, W = feats.size()
        seg_label = self._stack_batch_gt(batch_data_samples)  # (B, 1, orig_H, orig_W)

        # Downsample ground truth mask to match feature resolution
        seg_label = F.interpolate(seg_label.float(), size=(H, W), mode='nearest').long()

        # L2-normalize pixel feature vectors: ||z_i||_2 = 1
        feats_norm = F.normalize(feats, dim=1)

        # Flatten features to (N, C) and labels to (N,)
        feats_flat = feats_norm.permute(0, 2, 3, 1).contiguous().view(-1, C)
        labels_flat = seg_label.view(-1)

        # Update Prototypes via Exponential Moving Average (EMA)
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

        # If neither prototype is ready, return 0 loss
        if not self.proto_initialized.all():
            return {'loss_contrastive': feats.new_tensor(0.0)}

        # Normalized prototypes: (2, C)
        prototypes = F.normalize(self.prototypes, dim=1)

        # Compute cosine similarity between every pixel feature and dual prototypes:
        # proto_logits: (B, 2, H, W)
        proto_logits = torch.einsum('bchw, kc -> bkhw', feats_norm, prototypes) / self.temperature

        # InfoNCE loss: pulls pixels towards their class prototype and pushes away from other
        loss_proto = F.cross_entropy(proto_logits, seg_label.squeeze(1), ignore_index=self.ignore_index)

        # Prototype separation loss: encourage P_auth and P_tamp to point in opposite directions
        cos_sim = torch.dot(prototypes[0], prototypes[1])
        loss_sep = torch.clamp(cos_sim - self.sep_margin, min=0.0)

        total_loss = self.loss_weight_contrastive * loss_proto + self.loss_weight_sep * loss_sep
        return {'loss_contrastive': total_loss}
