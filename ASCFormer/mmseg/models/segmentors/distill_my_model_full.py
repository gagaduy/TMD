# Copyright (c) OpenMMLab. All rights reserved.
"""Cross-Modal / Teacher-Student Knowledge Distillation Segmentor for MiT-B0-Lite."""
import os
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from mmengine.config import Config

from mmseg.registry import MODELS
from mmseg.utils import SampleList, add_prefix
from .my_model_full import MyModelFull


def _replace_syncbn_cfg(node):
    if isinstance(node, dict):
        for k, v in node.items():
            if k == 'type' and v == 'SyncBN':
                node[k] = 'BN'
            else:
                _replace_syncbn_cfg(v)
    elif isinstance(node, list):
        for item in node:
            _replace_syncbn_cfg(item)


def _remap_nat_state_dict(state_dict: dict, target_keys: set) -> dict:
    """Auto-remap pure-PyTorch NAT keys (attn.qkv) <-> natten keys (attn.na.qkv)."""
    remapped = {}
    for k, v in state_dict.items():
        if k in target_keys:
            remapped[k] = v
            continue
        k_add_na = k.replace('.attn.qkv.', '.attn.na.qkv.').replace('.attn.proj.', '.attn.na.proj.')
        k_rm_na = k.replace('.attn.na.qkv.', '.attn.qkv.').replace('.attn.na.proj.', '.attn.proj.')
        if k_add_na in target_keys:
            remapped[k_add_na] = v
        elif k_rm_na in target_keys:
            remapped[k_rm_na] = v
        else:
            remapped[k] = v
    return remapped


@MODELS.register_module()
class MyModelFullDistill(MyModelFull):
    """Student MyModelFull with Foreground-Aware Knowledge Distillation from frozen Teacher MiT-B2.

    Teacher is stored in a non-Module list container `_teacher_holder` so:
      - Teacher weights are NEVER included in optimizer parameters or saved checkpoints
        (saving ~500 MB per checkpoint and ensuring 100% strict compatibility for ONNX export).
      - During val/test/export, zero Teacher overhead is executed.
    """

    def __init__(self,
                 teacher_cfg_path: Optional[str] = None,
                 teacher_ckpt_path: Optional[str] = None,
                 kd_temperature: float = 2.0,
                 kd_kl_weight: float = 1.0,
                 kd_dice_weight: float = 0.5,
                 kd_fg_weight: float = 5.0,
                 kd_soft_weight: float = 3.0,
                 **kwargs):
        super().__init__(**kwargs)
        self.kd_temperature = float(kd_temperature)
        self.kd_kl_weight = float(kd_kl_weight)
        self.kd_dice_weight = float(kd_dice_weight)
        self.kd_fg_weight = float(kd_fg_weight)
        self.kd_soft_weight = float(kd_soft_weight)
        self._teacher_holder = []
        self._teacher_device = None

        # Allow runtime override from environment variable (e.g., on Colab H100).
        # Setting RTM_TEACHER_CKPT="" or "NONE" explicitly disables Teacher loading.
        if 'RTM_TEACHER_CKPT' in os.environ:
            env_ckpt = os.environ['RTM_TEACHER_CKPT'].strip()
            teacher_ckpt_path = None if env_ckpt in ('', 'NONE', 'none', '0') else env_ckpt

        if teacher_cfg_path and teacher_ckpt_path and os.path.exists(teacher_ckpt_path):
            self._build_teacher(teacher_cfg_path, teacher_ckpt_path)
        elif teacher_ckpt_path:
            print(f'[KD] Teacher checkpoint not found at {teacher_ckpt_path}; running without Teacher KD.')

    def _build_teacher(self, cfg_path: str, ckpt_path: str) -> None:
        t_cfg = Config.fromfile(cfg_path)
        _replace_syncbn_cfg(t_cfg.model)
        for bk in ('backbone_main', 'backbone_extra', 'backbone_ocr'):
            if bk in t_cfg.model.backbone and 'pretrained' in t_cfg.model.backbone[bk]:
                t_cfg.model.backbone[bk].pretrained = None
                t_cfg.model.backbone[bk].init_cfg = None

        teacher = MODELS.build(t_cfg.model)
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        sd = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
        sd = _remap_nat_state_dict(sd, set(teacher.state_dict().keys()))
        missing, unexpected = teacher.load_state_dict(sd, strict=False)
        if missing:
            raise RuntimeError(f'[KD] Critical missing keys in Teacher checkpoint: {missing[:10]}')
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        self._teacher_holder = [teacher]
        print(f'>>>> [KD] Frozen Teacher loaded cleanly from: {ckpt_path} (ignored {len(unexpected)} aux keys) <<<<')

    def _get_teacher(self, device: torch.device):
        if not self._teacher_holder:
            return None
        teacher = self._teacher_holder[0]
        if self._teacher_device != device:
            teacher.to(device)
            teacher.eval()
            self._teacher_device = device
        return teacher

    def _compute_kd_loss(self, s_logits: Tensor, t_logits: Tensor, data_samples: SampleList) -> dict:
        # Operate in FP32 for numerical stability under AMP
        s_f = s_logits.float()
        t_f = t_logits.detach().float()
        if t_f.shape[2:] != s_f.shape[2:]:
            t_f = F.interpolate(t_f, size=s_f.shape[2:], mode='bilinear', align_corners=False)

        T = self.kd_temperature
        s_log_prob_T = F.log_softmax(s_f / T, dim=1)
        t_prob_T = F.softmax(t_f / T, dim=1)
        # Pixel-wise KL divergence: (B, H, W)
        kl_map = F.kl_div(s_log_prob_T, t_prob_T, reduction='none').sum(dim=1) * (T * T)

        # Build foreground + Teacher-confidence spatial weighting
        gt_list = [ds.gt_sem_seg.data for ds in data_samples]
        gt = torch.stack(gt_list, dim=0).to(s_f.device)  # (B, 1, H0, W0)
        gt_resized = F.interpolate(gt.float(), size=s_f.shape[2:], mode='nearest').squeeze(1)
        valid = (gt_resized != self.decode_head.ignore_index).float()
        fg_mask = (gt_resized == 1).float() * valid

        t_prob_1 = F.softmax(t_f, dim=1)[:, 1, :, :]  # (B, H, W)
        weights = (1.0 + self.kd_fg_weight * fg_mask + self.kd_soft_weight * t_prob_1) * valid
        loss_kl = (kl_map * weights).sum() / weights.sum().clamp(min=1.0)

        # Soft-Target Tampered Dice KD: transfers smooth tampered region localization
        s_prob_1 = F.softmax(s_f, dim=1)[:, 1, :, :] * valid
        t_prob_1_v = t_prob_1 * valid
        inter = (s_prob_1 * t_prob_1_v).sum(dim=(1, 2))
        denom = (s_prob_1.pow(2) + t_prob_1_v.pow(2)).sum(dim=(1, 2))
        loss_soft_dice = (1.0 - (2.0 * inter + 1.0) / (denom + 1.0)).mean()

        return {
            'kd.loss_kl': self.kd_kl_weight * loss_kl,
            'kd.loss_soft_dice': self.kd_dice_weight * loss_soft_dice,
        }

    def loss(self, inputs: Tensor, extras, data_samples: SampleList) -> dict:
        # Student forward (2-stream RGB + Forensic; never reads 'ocr')
        student_extras = {k: v for k, v in extras.items() if k != 'ocr'} if isinstance(extras, dict) else extras
        x = self.forward_encoder(inputs, student_extras)

        losses = dict()
        if hasattr(self.decode_head, 'loss_from_outputs'):
            seg_logits, feats = self.decode_head(x)
            loss_decode = self.decode_head.loss_from_outputs(seg_logits, feats, data_samples)
            losses.update(add_prefix(loss_decode, 'decode'))
        else:
            seg_logits = None
            losses.update(self._decode_head_forward_train(x, student_extras, data_samples))

        # Frozen Teacher forward for Foreground-Aware Knowledge Distillation
        teacher = self._get_teacher(inputs.device)
        if teacher is not None and seg_logits is not None:
            with torch.no_grad():
                t_extras = dict(extras) if isinstance(extras, dict) else extras
                if isinstance(t_extras, dict) and 'ocr' not in t_extras:
                    # Match Teacher B2's exact zero-OCR channel distribution
                    t_extras['ocr'] = torch.zeros_like(inputs)
                with torch.amp.autocast(device_type='cuda', enabled=inputs.is_cuda):
                    tx = teacher.forward_encoder(inputs, t_extras)
                    t_logits, _ = teacher.decode_head(tx)
            losses.update(self._compute_kd_loss(seg_logits, t_logits, data_samples))

        if self.with_auxiliary_head:
            losses.update(self._auxiliary_head_forward_train(x, data_samples))
        if self.with_extra_head:
            losses.update(self._extra_head_forward_train(x, student_extras))

        return losses
