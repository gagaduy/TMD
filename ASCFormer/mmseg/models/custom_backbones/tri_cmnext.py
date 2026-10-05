import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule
from mmseg.registry import MODELS
from ..utils import PatchEmbed, nchw_to_nlc, nlc_to_nchw
from ..toys.fusers import NATFuserBlock


@MODELS.register_module()
class TriCMNeXt(BaseModule):
    """Tri-Stream CMNeXt with Cascaded Cross-Attention for Real Text Manipulation Detection.

    Stream 1: Main Visual Branch (RGB) - MixVisionTransformer (MiT-B0)
    Stream 2: Forensic Auxiliary Branch (DCT, SRM, ELA) - HubVisionTransformer (MiT-B0)
    Stream 3: OCR Spatial/Geometric Branch (Region, Boundary, Distance) - MixVisionTransformer (MiT-B0)

    Fusion at each stage i:
        1. Forensic calibration: x_fused_ra = FFMs[i](x_cam, x_f)
        2. OCR structural calibration: x_fused_final = FFM_ocrs[i](x_fused_ra, x_ocr)
    """

    def __init__(self,
                 backbone_main: dict,
                 backbone_extra: dict,
                 backbone_ocr: dict,
                 fuser: dict = None,
                 fuser_ocr: dict = None,
                 num_heads=[1, 2, 5, 8],
                 in_stages=None,
                 extra_patch_embed: dict = None,
                 out_indices=(0, 1, 2, 3),
                 use_rectifier: bool = False,
                 rectifier=None,
                 init_cfg=None,
                 no_select=False,
                 **kwargs):
        super().__init__(init_cfg=init_cfg)

        self.out_indices = out_indices
        self.in_stages = in_stages
        self.no_select = no_select

        # Build 3 Stream Backbones
        self.main_branch = MODELS.build(backbone_main)
        self.extra_branch = MODELS.build(backbone_extra)
        self.ocr_branch = MODELS.build(backbone_ocr)

        if in_stages is not None:
            assert len(in_stages) == self.extra_branch.num_modals

        if extra_patch_embed is not None:
            self.extra_patch_embed = PatchEmbed(
                in_channels=extra_patch_embed['in_channels'],
                embed_dims=extra_patch_embed['embed_dims'],
                kernel_size=extra_patch_embed['kernel_size'],
                stride=extra_patch_embed['stride'],
                padding=extra_patch_embed['kernel_size'] // 2,
                norm_cfg=dict(type='LN', eps=1e-6),
            )
            self.use_extra_patch_embed = True
            self.reshape_extra_nchw = extra_patch_embed.get('reshape', True)
        else:
            self.use_extra_patch_embed = False

        self.num_stage_main = len(self.main_branch.num_layers)
        self.num_stage_extra = len(self.extra_branch.num_layers)
        self.num_stage_ocr = len(self.ocr_branch.num_layers)
        assert self.num_stage_main >= self.num_stage_extra
        assert self.num_stage_main == self.num_stage_ocr
        self.shift_stage = self.num_stage_main - self.num_stage_extra

        num_heads = self.extra_branch.num_heads
        embed_dims = [self.extra_branch.embed_dims * num_heads[i] for i in range(len(num_heads))]

        # Stage fusers:
        # FFM: RGB + Forensic
        self.FFMs = nn.ModuleList()
        # FFM_ocr: [RGB+Forensic] + OCR
        self.FFM_ocrs = nn.ModuleList()

        for i in range(len(num_heads)):
            # Forensic fuser
            cfg_fuser = dict(
                type='NATFuserBlock',
                a_channel=embed_dims[i],
                b_channel=embed_dims[i],
                num_head=self.extra_branch.num_heads[i],
                kernel_size=5,
                gated=True,
                post_attn=True,
                attn_mode='cross'
            )
            if fuser is not None:
                cfg_fuser.update(fuser)
                cfg_fuser.update(dict(a_channel=embed_dims[i], b_channel=embed_dims[i], num_head=self.extra_branch.num_heads[i]))
            self.FFMs.append(MODELS.build(cfg_fuser))

            # OCR fuser
            cfg_ocr = dict(
                type='NATFuserBlock',
                a_channel=embed_dims[i],
                b_channel=embed_dims[i],
                num_head=self.extra_branch.num_heads[i],
                kernel_size=5,
                gated=True,
                post_attn=True,
                attn_mode='cross'
            )
            if fuser_ocr is not None:
                cfg_ocr.update(fuser_ocr)
                cfg_ocr.update(dict(a_channel=embed_dims[i], b_channel=embed_dims[i], num_head=self.extra_branch.num_heads[i]))
            self.FFM_ocrs.append(MODELS.build(cfg_ocr))

    def forward(self, x):
        outs = []
        x_cam = x[0]    # RGB [B, 3, H, W]
        x_extra = x[1]  # Forensic list [dct, ela, srm]
        x_ocr = x[2] if len(x) > 2 else None  # OCR map [B, 3, H, W]

        # align channel for extra modalities
        for i in range(len(x_extra)):
            if x_extra[i].size()[1] != self.extra_branch.in_channels:
                if (x_extra[i].size()[1] == 1) & (x_cam.size()[2:] == x_extra[i].size()[2:]):
                    x_extra[i] = x_extra[i].repeat(1, self.extra_branch.in_channels, 1, 1)

        if self.in_stages is not None and max(self.in_stages) > 0:
            in_stage = max(self.in_stages)
            x_extra_0 = []
            x_extra_m = []
            for i in range(len(self.in_stages)):
                if self.in_stages[i] > 0:
                    x_extra_m.append(x_extra[i])
                else:
                    x_extra_0.append(x_extra[i])
            x_extra = x_extra_0
        else:
            in_stage = -1

        B = x_cam.shape[0]

        # Shift stages if main has more stages than extra
        for i in range(self.shift_stage):
            layer = self.main_branch.layers[i]
            x_cam, hw_shape = layer[0](x_cam)
            for block in layer[1]:
                x_cam = block(x_cam, hw_shape)
            x_cam = layer[2](x_cam)
            x_cam = nlc_to_nchw(x_cam, hw_shape)
            if i in self.out_indices:
                outs.append(x_cam)

        # Multi-stage Tri-modal forward and cascaded fusion
        for i in range(self.num_stage_extra):
            # 1. Main visual stream (RGB)
            layer = self.main_branch.layers[(i + self.shift_stage)]
            x_cam, hw_shape = layer[0](x_cam)
            H, W = hw_shape
            for block in layer[1]:
                x_cam = block(x_cam, hw_shape)
            x_cam = layer[2](x_cam)
            x_cam = nlc_to_nchw(x_cam, hw_shape)

            # 2. Forensic auxiliary stream (DCT, ELA, SRM)
            layer_extra = self.extra_branch.layers[i]
            if self.in_stages is not None and i == in_stage:
                if self.use_extra_patch_embed:
                    if self.reshape_extra_nchw:
                        temp = []
                        for item in x_extra_m:
                            em, s = self.extra_patch_embed(item)
                            temp.append(nlc_to_nchw(em, s))
                        x_extra_m = temp
                    else:
                        x_extra_m = [self.extra_patch_embed(item)[0] for item in x_extra_m]
                    x_extra, _ = layer_extra[0](x_extra)
                    x_extra.extend(x_extra_m)
                else:
                    x_extra.extend(x_extra_m)
                    x_extra, _ = layer_extra[0](x_extra)
            else:
                x_extra, _ = layer_extra[0](x_extra)

            if self.no_select:
                x_f = torch.stack(x_extra, dim=0).mean(dim=0)
                x_f = x_f.flatten(2).transpose(1, 2)
            else:
                x_f = self.extra_branch.tokenselect(x_extra, self.extra_branch.extra_score_predictor[i])

            for block in layer_extra[1]:
                x_f = block(x_f, hw_shape)
            x_f = layer_extra[2](x_f)
            x_f = nlc_to_nchw(x_f, hw_shape)

            # Stage 1 Fusion: RGB + Forensic -> Anomalous tampering features
            x_fused_ra = self.FFMs[i](x_cam, x_f)

            # 3. OCR stream (if provided)
            if x_ocr is not None:
                layer_ocr = self.ocr_branch.layers[(i + self.shift_stage)]
                x_ocr_t, hw_shape_ocr = layer_ocr[0](x_ocr)
                for block in layer_ocr[1]:
                    x_ocr_t = block(x_ocr_t, hw_shape_ocr)
                x_ocr_t = layer_ocr[2](x_ocr_t)
                x_ocr_feat = nlc_to_nchw(x_ocr_t, hw_shape_ocr)

                # Stage 2 Fusion: [RGB+Forensic] + OCR -> Layout-guided & Boundary-sharpened features
                x_fused_final = self.FFM_ocrs[i](x_fused_ra, x_ocr_feat)
                x_ocr = x_ocr_feat
            else:
                x_fused_final = x_fused_ra

            if (i + self.shift_stage) in self.out_indices:
                outs.append(x_fused_final)

            x_extra = [x_.reshape(B, H, W, -1).permute(0, 3, 1, 2) + x_f for x_ in x_extra] if self.extra_branch.num_modals > 1 else [x_f]

        return outs
