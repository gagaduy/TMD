"""Controlled Ablation G4 (16k x bs=8 = 128k crops, lr=6e-5, PolyLR) — M2: M1 + Real OCR 3-Stream.

Isolates Contribution OCR ONLY (with fixed ocr_dir path loading real PaddleOCR maps BEFORE RandomFlip
so H/V flips align 100% with img & gt_seg_map). Keeps RandomCrop and ContrastiveHeadV2 identical to M1.
"""
_base_ = ['./ascformer_rtm_b0_tristream.py']

randomness = dict(seed=3407)
crop_size = (512, 512)
quality = 80

model = dict(
    backbone=dict(
        backbone_main=dict(drop_path_rate=0.1),
        backbone_extra=dict(drop_path_rate=0.1),
        backbone_ocr=dict(drop_path_rate=0.1),
    ),
    test_cfg=dict(mode='slide', crop_size=(512, 512), stride=(384, 384)),
)

# Load real OCR spatial map BEFORE RandomFlip so both H & V flips synchronize automatically
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', binary=True),
    dict(type='LoadOCRSpatialFromFile', ocr_dir='data/ttd/RealTextMan/ocr_spatial'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='RandomFlip', prob=0.5, direction='vertical'),
    dict(
        type='RandomCropWithExtra',
        crop_size=crop_size,
        stride=8,
        extra_keys=('ocr',),
        cat_max_ratio=0.75),
    dict(type='ELA', quality=quality),
    dict(type='BlockDCT', zigzag=True),
    dict(type='PackSegInputsWithExtra', extra_keys=('dct', 'ela', 'ocr')),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='ELA', quality=quality),
    dict(type='BlockDCT', zigzag=True),
    dict(type='LoadAnnotations', binary=True),
    dict(type='LoadOCRSpatialFromFile', ocr_dir='data/ttd/RealTextMan/ocr_spatial'),
    dict(type='PackSegInputsWithExtra', extra_keys=('dct', 'ela', 'ocr')),
]

train_dataloader = dict(
    batch_size=8,
    num_workers=3,
    persistent_workers=True,
    pin_memory=True,
    dataset=dict(pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=1,
    num_workers=3,
    persistent_workers=True,
    pin_memory=True,
    dataset=dict(pipeline=test_pipeline))
test_dataloader = val_dataloader

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
            'fuser_ocr': dict(lr_mult=5.),
            'rpb': dict(decay_mult=0.),
        }))

train_cfg = dict(type='IterBasedTrainLoop', max_iters=16000, val_interval=8000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1000),
    dict(type='PolyLR', eta_min=0.0, power=1.0, by_epoch=False, begin=1000, end=16000),
]

val_evaluator = dict(
    type='BinaryIoUMetric',
    iou_metrics=['mIoU', 'mDice', 'mFscore'],
    output_dir='work_dirs/ablation_b0_16k_bs8_m2/val_metrics',
    save_confusion_matrix=True)
test_evaluator = val_evaluator

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=200, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=4000,
        max_keep_ckpts=2,
        save_best='mIoU',
        rule='greater'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))

find_unused_parameters = True
