"""Tri-Stream MiT-B0 configuration with OCR Spatial Guidance (Region, Boundary, Distance)."""
_base_ = ['./ascformer_rtm_b0.py']

checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b0_20220624-7e0fe6dd.pth'
crop_size = (512, 512)
quality = 80

# Update Model Backbone to TriCMNeXt with 3rd OCR Stream
model = dict(
    backbone=dict(
        type='TriCMNeXt',
        backbone_ocr=dict(
            type='MixVisionTransformer',
            pretrained=checkpoint,
            in_channels=3,
            embed_dims=32,
            num_stages=4,
            num_layers=[2, 2, 2, 2],
            num_heads=[1, 2, 5, 8],
            patch_sizes=[7, 3, 3, 3],
            sr_ratios=[8, 4, 2, 1],
            out_indices=(0, 1, 2, 3),
            mlp_ratio=4,
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.1),
        fuser_ocr=dict(
            type='NATFuserBlock',
            kernel_size=5,
            gated=True,
            post_attn=True,
            attn_mode='cross',
        ),
    )
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', binary=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='RandomFlip', prob=0.5, direction='vertical'),
    dict(type='ELA', quality=quality),
    dict(type='BlockDCT', zigzag=True),
    dict(type='LoadOCRSpatialFromFile'),
    dict(type='RandomCropWithExtra',
         crop_size=crop_size,
         stride=8,
         extra_keys=('dct', 'ela', 'ocr'),
         cat_max_ratio=0.75),
    dict(type='PackSegInputsWithExtra', extra_keys=('dct', 'ela', 'ocr'))
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='ELA', quality=quality),
    dict(type='BlockDCT', zigzag=True),
    dict(type='LoadAnnotations', binary=True),
    dict(type='LoadOCRSpatialFromFile'),
    dict(type='PackSegInputsWithExtra', extra_keys=('dct', 'ela', 'ocr'))
]

train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader = dict(dataset=dict(pipeline=test_pipeline))
