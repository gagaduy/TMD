_base_ = './ascformer_rtm.py'

norm_cfg = dict(type='SyncBN', requires_grad=True)
crop_size = (512, 512)
quality = 80

model = dict(
    backbone=dict(
        in_stages=(1, 0, 0, 0),
        backbone_extra=dict(
            modals=['dct', 'srm', 'ela', 'dwt'],
            in_modals=(3, 4, 4, 4),
        ),
    ),
    preprocessor_sec=[
        [
            'dct',
            dict(
                type='DCTProcessor',
                in_channels=1,
                embed_dims=64,
                num_heads=1,
                patch_size=3,
                stride=1,
                sr_ratio=4,
                out_channels=64,
                norm_cfg=norm_cfg,
                reduce_neg=False,
            )
        ],
        ['ela', dict(type='NoFilter')],
        [
            'img',
            dict(
                type='SRMConv2d_simple',
                inc=3,
                learnable=False,
            ),
        ],
        ['dwt', dict(type='NoFilter')],
    ],
)

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

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='ELA', quality=quality),
    dict(type='BlockDCT', zigzag=True),
    dict(type='HaarDWT'),
    dict(type='LoadAnnotations', binary=True),
    dict(type='PackSegInputsWithExtra', extra_keys=('dct', 'ela', 'dwt')),
]

train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader = dict(dataset=dict(pipeline=test_pipeline))
