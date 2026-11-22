"""Full-Factorial 2^3 Ablation G4 (16k x bs=8, lr=6e-5, PolyLR) — M6: Real OCR + BalancedPrototypeHead.

Factor matrix: OCR=True | Focused+Rotate90=False | BalancedPrototypeHead=True.
"""
_base_ = ['./ablation_b0_16k_bs8_m2_ocr.py']

norm_cfg = dict(type='SyncBN', requires_grad=True)

model = dict(
    decode_head=dict(
        _delete_=True,
        type='BalancedPrototypeHead',
        in_channels=[32, 64, 160, 256],
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        up_decode=False,
        dim=128,
        temperature=0.1,
        momentum=0.99,
        loss_weight_contrastive=0.05,
        sep_margin=0.0,
        loss_weight_sep=0.05,
        proto_fuse_weight=0.25,
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
)

val_evaluator = dict(output_dir='work_dirs/ablation_b0_16k_bs8_m6_bal/val_metrics')
test_evaluator = val_evaluator
