"""FINAL PAPER-GRADE CHAMPION CONFIG (960k crops = 80,000 iters x bs=12, NO KD):
MiT-B0 2-Stream + FocusedCrop(0.65) + RandomRotate90(0.5) + BalancedPrototypeHead-256.

Winner of Full-Factorial 2^3 Ablation (M7):
  - #1/8 at BOTH tau=0.50 (8.38% IoU, 15.47% F1, 45.43% Prec) and Peak tau*=0.20 (12.44% IoU, 22.12% F1) at 16k.
  - 15.17M params, 72.35 GFLOPs (3.2x fewer FLOPs than M3, 5.3x fewer than MiT-B2), ~3.6s Full-Scan CPU (8 tiles).
  - Exact 960,000-crop budget matching ASC-Former (Pattern Recognition 2025) Table 4:
      80,000 iters x batch_size 12 = 960,000 crops (~165.5 epochs), AdamW lr=6e-5, Warmup 1.5k + PolyLR, BF16.
"""
_base_ = ['./ablation_b0_16k_bs8_m7_focused_balproto.py']

train_dataloader = dict(
    batch_size=12,
    num_workers=8,
    persistent_workers=True,
    pin_memory=True,
)

val_dataloader = dict(
    batch_size=1,
    num_workers=6,
    persistent_workers=True,
    pin_memory=True,
)
test_dataloader = val_dataloader

# Exact proven lr=6e-5 (matching paper 80k x 12 and M7 16k x 8) with BF16 + grad clip
optim_wrapper = dict(
    _delete_=True,
    type='AmpOptimWrapper',
    dtype='bfloat16',
    loss_scale=dict(enabled=False),
    optimizer=dict(type='AdamW', lr=6e-5, betas=(0.9, 0.999), weight_decay=0.01),
    clip_grad=dict(max_norm=1.0),
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.),
            'rpb': dict(decay_mult=0.),
        }))

# 80,000 iters x batch_size 12 = 960,000 crops (100% matched to ASC-Former paper Table 4)
train_cfg = dict(type='IterBasedTrainLoop', max_iters=80000, val_interval=10000)

param_scheduler = [
    dict(type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(type='PolyLR', eta_min=0.0, power=1.0, by_epoch=False, begin=1500, end=80000),
]

val_evaluator = dict(
    type='BinaryIoUMetric',
    iou_metrics=['mIoU', 'mDice', 'mFscore'],
    output_dir='work_dirs/final_b0_m7_balproto_960k/val_metrics',
    save_confusion_matrix=True)
test_evaluator = val_evaluator

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=200, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=10000,
        max_keep_ckpts=3,
        save_best='mIoU',
        rule='greater'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))
