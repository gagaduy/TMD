"""Single-GPU Tri-Stream MiT-B0 40k Training with Dual-Prototype Contrastive Learning + Focused Crop."""
_base_ = ['./ascformer_rtm_b0_tristream_proto_pilot8k.py']

# Scale batch size to 4 for faster convergence and efficient GPU tensor operations
train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True
)

# 40,000 iterations (~27.6 epochs) with periodic validation and checkpoint saving
train_cfg = dict(max_iters=40000, val_interval=10000)

param_scheduler = [
    dict(type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(type='PolyLR', eta_min=0., power=1., by_epoch=False,
         begin=1500, end=40000),
]

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=5000,
        max_keep_ckpts=5,
        save_best='mIoU',
        rule='greater'
    )
)

val_evaluator = dict(
    iou_metrics=['mIoU', 'mDice', 'mFscore'],
    output_dir='work_dirs/tristream_proto_40k/val_metrics',
    save_confusion_matrix=True
)
