"""Ablation 16k - M4: Tri-Stream + Focused Sampling + Dual-Prototype Contrastive Head MiT-B0 on A100."""
_base_ = ['./ascformer_rtm_b0_ablation16k_m3_focused.py']

norm_cfg = dict(type='SyncBN', requires_grad=True)

model = dict(
    decode_head=dict(
        _delete_=True,
        type='PrototypeContrastiveHead',
        in_channels=[32, 64, 160, 256],
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        up_decode=True,
        dim=128,
        temperature=0.1,
        momentum=0.99,
        loss_weight_contrastive=0.05,
        sep_margin=0.0,
        loss_weight_sep=0.01,
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)
    )
)

val_evaluator = dict(
    type='BinaryIoUMetric',
    iou_metrics=['mIoU', 'mDice', 'mFscore'],
    output_dir='work_dirs/ablation16k_m4_proto/val_metrics',
    save_confusion_matrix=True
)
