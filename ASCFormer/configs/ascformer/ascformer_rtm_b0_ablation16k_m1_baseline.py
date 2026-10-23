"""Ablation 16k - M1: Baseline 2-Stream (RGB + Forensic Hub) MiT-B0 on A100 (Parallel Mode)."""
_base_ = ['./ascformer_rtm_b0.py']

randomness = dict(seed=3407)
crop_size = (512, 512)
quality = 80

# Crop 512x512 at stride=8 (8x8 JPEG block aligned) BEFORE ELA & BlockDCT -> 15x faster CPU pipeline!
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', binary=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='RandomFlip', prob=0.5, direction='vertical'),
    dict(type='RandomCropWithExtra',
         crop_size=crop_size,
         stride=8,
         extra_keys=(),
         cat_max_ratio=0.75),
    dict(type='ELA', quality=quality),
    dict(type='BlockDCT', zigzag=True),
    dict(type='PackSegInputsWithExtra', extra_keys=('dct', 'ela'))
]

train_dataloader = dict(
    batch_size=4,
    num_workers=3,
    persistent_workers=True,
    pin_memory=True,
    dataset=dict(pipeline=train_pipeline)
)
val_dataloader = dict(
    batch_size=1,
    num_workers=3,
    persistent_workers=True,
    pin_memory=True
)
test_dataloader = dict(
    batch_size=1,
    num_workers=3,
    persistent_workers=True,
    pin_memory=True
)

optim_wrapper = dict(
    type='AmpOptimWrapper',
    loss_scale='dynamic',
    optimizer=dict(type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.),
            'rpb': dict(decay_mult=0.),
        }
    )
)

train_cfg = dict(max_iters=16000, val_interval=16000)

param_scheduler = [
    dict(type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1000),
    dict(type='PolyLR', eta_min=0., power=1., by_epoch=False, begin=1000, end=16000),
]

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=16000, max_keep_ckpts=1, save_best=None)
)

val_evaluator = dict(
    type='BinaryIoUMetric',
    iou_metrics=['mIoU', 'mDice', 'mFscore'],
    output_dir='work_dirs/ablation16k_m1_baseline/val_metrics',
    save_confusion_matrix=True
)

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer')
