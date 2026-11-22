"""Controlled Ablation G4 (16k x bs=8 = 128k crops, lr=6e-5, PolyLR) — M4: M1 + LitePrototypeHead.

Isolates Contribution Lightweight Dual-Prototype Decode Head ONLY
(2-Stream AsymCMNeXtV2, NO OCR, keeps RandomCrop & pure CrossEntropyLoss identical to M1;
cuts head FLOPs by 97% to 5.90 GFLOPs. Per-image DiceLoss removed because it collapses logits
to ~-8 on pure-background RandomCrops).
"""
_base_ = ['./ablation_b0_16k_bs8_m1_baseline.py']

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
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
)

val_evaluator = dict(output_dir='work_dirs/ablation_b0_16k_bs8_m4/val_metrics')
test_evaluator = val_evaluator
