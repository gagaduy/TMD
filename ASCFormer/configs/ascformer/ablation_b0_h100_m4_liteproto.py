"""Controlled Ablation H100 (10k x bs=16 = 160k crops) — M4: M1 + LitePrototypeHead (DW-Sep + CE+Dice).

Isolates Contribution Lightweight Dual-Prototype Decode Head & Compound Loss ONLY
(2-Stream AsymCMNeXtV2, NO OCR, keeps RandomCrop identical to M1; cuts head FLOPs by 97% to 5.90 GFLOPs).
"""
_base_ = ['./ablation_b0_h100_m1_baseline.py']

norm_cfg = dict(type='SyncBN', requires_grad=True)

model = dict(
    decode_head=dict(
        _delete_=True,
        type='LitePrototypeHead',
        in_channels=[32, 64, 160, 256],
        in_index=[0, 1, 2, 3],
        channels=128,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        up_decode=True,
        dim=64,
        temperature=0.1,
        momentum=0.99,
        loss_weight_contrastive=0.05,
        sep_margin=0.0,
        loss_weight_sep=0.01,
        proto_fuse_weight=0.5,
        loss_decode=[
            dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            dict(type='DiceLoss', use_sigmoid=False, loss_weight=0.5, eps=1e-5, ignore_index=255),
        ]),
)

val_evaluator = dict(output_dir='work_dirs/ablation_b0_h100_m4/val_metrics')
test_evaluator = val_evaluator
