_base_ = './ascformer_rtm_dwt.py'

experiment_split = 'experiment_splits/seed_20260903'
crop_size = (384, 384)
quality = 80

data_preprocessor = dict(size=crop_size)
model = dict(data_preprocessor=dict(size=crop_size))

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', binary=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='RandomFlip', prob=0.5, direction='vertical'),
    dict(type='ELA', quality=quality),
    dict(type='BlockDCT', zigzag=True),
    dict(type='HaarDWT'),
    dict(
        type='RandomCropWithExtra',
        crop_size=crop_size,
        stride=8,
        extra_keys=('dct', 'ela', 'dwt'),
        cat_max_ratio=0.75),
    dict(type='PackSegInputsWithExtra', extra_keys=('dct', 'ela', 'dwt')),
]

train_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    dataset=dict(ann_file=f'{experiment_split}/train.txt', pipeline=train_pipeline))
val_dataloader = dict(dataset=dict(ann_file=f'{experiment_split}/val.txt'))
test_dataloader = dict(dataset=dict(ann_file='test.txt'))
