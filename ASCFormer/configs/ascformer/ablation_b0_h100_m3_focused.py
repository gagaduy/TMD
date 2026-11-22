"""Controlled Ablation H100 (10k x bs=16 = 160k crops) — M3: M1 + FocusedCrop(0.65) + RandomRotate90.

Isolates Contribution Data Sampling & Forensic-Safe Augmentation ONLY (2-Stream AsymCMNeXtV2,
NO OCR, keeps ContrastiveHeadV2 Memory Bank identical to M1).
"""
_base_ = ['./ablation_b0_h100_m1_baseline.py']

crop_size = (512, 512)
quality = 80
pos_prob = 0.65

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', binary=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='RandomFlip', prob=0.5, direction='vertical'),
    dict(type='RandomRotate90', prob=0.5),
    dict(
        type='FocusedCropWithExtra',
        crop_size=crop_size,
        stride=8,
        pos_prob=pos_prob,
        extra_keys=()),
    dict(type='ELA', quality=quality),
    dict(type='BlockDCT', zigzag=True),
    dict(type='PackSegInputsWithExtra', extra_keys=('dct', 'ela')),
]

train_dataloader = dict(dataset=dict(pipeline=train_pipeline))

val_evaluator = dict(output_dir='work_dirs/ablation_b0_h100_m3/val_metrics')
test_evaluator = val_evaluator
