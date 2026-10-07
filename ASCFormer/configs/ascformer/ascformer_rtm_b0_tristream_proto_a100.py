"""High-throughput Tri-Stream MiT-B0 pilot for NVIDIA A100 GPU (~10-12 minutes).

Batch size 8 with 2,000 iterations processes exactly 16,000 samples (equivalent
data budget to 8,000 iterations at batch size 2), fully utilizing A100 VRAM
and Tensor Cores while finishing in ~10 minutes.
"""
_base_ = ['./ascformer_rtm_b0_tristream_proto_pilot8k.py']

# Scale batch size and workers for A100 SXM4 (40GB VRAM + high vCPU count)
train_dataloader = dict(
    batch_size=8,
    num_workers=8,
    persistent_workers=True
)

# 2000 iters * batch_size 8 = 16,000 samples (identical to 8000 iters * batch_size 2)
train_cfg = dict(max_iters=2000, val_interval=2000)

# Warmup 200 iters + PolyLR decay up to 2000 iters
param_scheduler = [
    dict(type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=200),
    dict(type='PolyLR', eta_min=0., power=1., by_epoch=False, begin=200, end=2000),
]

# Scaled learning rate for 4x larger batch size (linear scaling rule)
optim_wrapper = dict(
    type='AmpOptimWrapper',
    loss_scale='dynamic',
    optimizer=dict(
        type='AdamW',
        lr=0.00012,
        betas=(0.9, 0.999),
        weight_decay=0.01
    )
)

default_hooks = dict(
    checkpoint=dict(
        interval=2000,
        max_keep_ckpts=1,
        save_best=None
    )
)

val_evaluator = dict(
    iou_metrics=['mIoU', 'mDice', 'mFscore'],
    output_dir='work_dirs/a100_b0_tristream_proto/val_metrics',
    save_confusion_matrix=True
)
