"""Single-GPU Tri-Stream MiT-B0 pilot (8k iters) with Dual-Prototype Contrastive Learning + Focused Crop."""
_base_ = ['./ascformer_rtm_b0_tristream_focusedcrop_pilot8k.py']

norm_cfg = dict(type='SyncBN', requires_grad=True)

model = dict(
    decode_head=dict(
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
    iou_metrics=['mIoU', 'mDice', 'mFscore'],
    output_dir='work_dirs/pilot8k_b0_tristream_proto/val_metrics',
    save_confusion_matrix=True)
