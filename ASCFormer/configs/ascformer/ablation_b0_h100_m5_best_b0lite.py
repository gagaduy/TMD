"""Controlled Ablation H100 (10k x bs=16 = 160k crops) — M5: M3 + M4 (Best Optimized MiT-B0-Lite).

Combines the winning non-OCR components:
  - 2-Stream AsymCMNeXtV2 MiT-B0 (13.74M params, 29.07 GFLOPs, 30.35 MB INT8 ONNX, ~378 ms/tile CPU)
  - FocusedCrop (pos_prob=0.65, stride=8) + RandomRotate90 + H/V Flip
  - LitePrototypeHead (channels=128, dim=64, DW-Sep 3x3, Dual EMA Prototypes, CE + Dice Loss)
"""
_base_ = ['./ablation_b0_h100_m4_liteproto.py']

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

val_evaluator = dict(output_dir='work_dirs/ablation_b0_h100_m5/val_metrics')
test_evaluator = val_evaluator
