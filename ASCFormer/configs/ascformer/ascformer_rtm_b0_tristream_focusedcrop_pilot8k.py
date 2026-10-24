"""Single-GPU Tri-Stream MiT-B0 pilot (8k iters) with Focused Positive-Biased Cropping."""
_base_ = ['./ascformer_rtm_b0_tristream.py']

randomness = dict(seed=3407)
pos_prob = 0.7
crop_size = (512, 512)
quality = 80

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', binary=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='RandomFlip', prob=0.5, direction='vertical'),
    dict(type='ELA', quality=quality),
    dict(type='BlockDCT', zigzag=True),
    dict(type='LoadOCRSpatialFromFile'),
    dict(type='FocusedCropWithExtra',
         crop_size=crop_size,
         stride=8,
         pos_prob=pos_prob,
         extra_keys=('dct', 'ela', 'ocr')),
    dict(type='PackSegInputsWithExtra', extra_keys=('dct', 'ela', 'ocr'))
]

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    dataset=dict(pipeline=train_pipeline)
)
val_dataloader = dict(batch_size=1, num_workers=2, persistent_workers=True)

optim_wrapper = dict(type='AmpOptimWrapper', loss_scale='dynamic')
train_cfg = dict(max_iters=8000, val_interval=8000)
param_scheduler = [
    dict(type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=800),
    dict(type='PolyLR', eta_min=0., power=1., by_epoch=False,
         begin=800, end=8000),
]
default_hooks = dict(checkpoint=dict(
    interval=8000, max_keep_ckpts=1, save_best=None))
val_evaluator = dict(
    iou_metrics=['mIoU', 'mDice', 'mFscore'],
    output_dir='work_dirs/pilot8k_b0_tristream_focused/val_metrics',
    save_confusion_matrix=True)
