"""Full-Factorial 2^3 Ablation G4 (16k x bs=8, lr=6e-5, PolyLR) — M5: OCR + FocusedCrop(0.65) + Rotate90.

Factor matrix: OCR=True | Focused+Rotate90=True | LitePrototypeHead=False (keeps ContrastiveHeadV2).
"""
_base_ = ['./ablation_b0_16k_bs8_m2_ocr.py']

crop_size = (512, 512)
quality = 80
pos_prob = 0.65

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', binary=True),
    dict(type='LoadOCRSpatialFromFile', ocr_dir='data/ttd/RealTextMan/ocr_spatial'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='RandomFlip', prob=0.5, direction='vertical'),
    dict(type='RandomRotate90', prob=0.5),
    dict(
        type='FocusedCropWithExtra',
        crop_size=crop_size,
        stride=8,
        pos_prob=pos_prob,
        extra_keys=('ocr',)),
    dict(type='ELA', quality=quality),
    dict(type='BlockDCT', zigzag=True),
    dict(type='PackSegInputsWithExtra', extra_keys=('dct', 'ela', 'ocr')),
]

train_dataloader = dict(dataset=dict(pipeline=train_pipeline))

val_evaluator = dict(output_dir='work_dirs/ablation_b0_16k_bs8_m5/val_metrics')
test_evaluator = val_evaluator
