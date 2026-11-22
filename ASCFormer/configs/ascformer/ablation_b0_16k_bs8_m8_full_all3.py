"""Full-Factorial 2^3 Ablation G4 (16k x bs=8, lr=6e-5, PolyLR) — M8: OCR + FocusedCrop + Rotate90 + LiteProto.

Factor matrix: OCR=True | Focused+Rotate90=True | LitePrototypeHead=True (pure CE).
"""
_base_ = ['./ablation_b0_16k_bs8_m5_ocr_focused.py']

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

val_evaluator = dict(output_dir='work_dirs/ablation_b0_16k_bs8_m8/val_metrics')
test_evaluator = val_evaluator
