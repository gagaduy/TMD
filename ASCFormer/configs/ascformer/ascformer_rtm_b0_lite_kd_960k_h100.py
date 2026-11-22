"""Optimized MiT-B0-Lite with Cross-Modal Knowledge Distillation from Teacher MiT-B2 SOTA.

Training Budget: 60,000 iterations x batch_size 16 = 960,000 crops (exact paper budget on Colab H100).
Key Optimizations specifically tailored for MiT-B0 Mobile Full-Scan (No Early Exit):
  1. 2-Stream AsymCMNeXtV2 (RGB + Forensic Hub [DCT, SRM, ELA]) — removes redundant 3rd Transformer
     stream and eliminates 100% of Train-Deploy OCR domain mismatch.
  2. LitePrototypeHead (channels=128, dim=64, 1x1 Fusion + Depthwise-Separable 3x3) — eliminates the
     83.9% decoder FLOPs bottleneck (drops total GFLOPs from 231.5 -> ~21 GFLOPs/tile).
  3. Compound Loss: CrossEntropyLoss + DiceLoss + Dual EMA Prototype InfoNCE + Separation Loss.
  4. Foreground-Aware Knowledge Distillation (Weighted KL + Soft Dice KD) from frozen Teacher MiT-B2 SOTA
     (19.91% Tamper IoU / 33.20% F1).
  5. Forensic-Safe Augmentations (H/V Flip + RandomRotate90 preserving 8x8 DCT grid) + FocusedCrop (0.65)
     + DCT-aligned sliding-window evaluation stride=(384, 384).
"""

_base_ = [
    '../_base_/datasets/rtm_crop.py',
    '../_base_/default_runtime.py',
]

randomness = dict(seed=3407)
norm_cfg = dict(type='SyncBN', requires_grad=True)
crop_size = (512, 512)
num_classes = 2
quality = 80
pos_prob = 0.65

checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b0_20220624-7e0fe6dd.pth'

data_preprocessor = dict(
    type='SegDataPreProcessorWithExtra',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    size=crop_size,
    pad_val=0,
    seg_pad_val=255,
    copy_img=True,
    test_cfg=dict(size_divisor=32))

model = dict(
    type='MyModelFullDistill',
    teacher_cfg_path='configs/ascformer/ascformer_rtm_b2_tristream_proto_60k_a100.py',
    teacher_ckpt_path='work_dirs/full30k_b2_sota/best_mIoU_iter_30000.pth',
    kd_temperature=2.0,
    kd_kl_weight=1.0,
    kd_dice_weight=0.5,
    kd_fg_weight=5.0,
    kd_soft_weight=3.0,
    merge_input=True,
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='AsymCMNeXtV2',
        use_rectifier=False,
        in_stages=(1, 0, 0),
        extra_patch_embed=dict(
            in_channels=32,
            embed_dims=64,
            kernel_size=3,
            stride=1,
            reshape=True),
        out_indices=(0, 1, 2, 3),
        backbone_main=dict(
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
            drop_path_rate=0.05),
        backbone_extra=dict(
            type='HubVisionTransformer',
            pretrained=checkpoint,
            in_channels=3,
            embed_dims=32,
            modals=['dct', 'srm', 'ela'],
            in_modals=(2, 3, 3, 3),
            skip_patch_embed_stage=1,
            num_stages=4,
            num_layers=[2, 2, 2, 2],
            num_heads=[1, 2, 5, 8],
            patch_sizes=[7, 3, 3, 3],
            strides=[4, 2, 2, 2],
            sr_ratios=[8, 4, 2, 1],
            out_indices=(0, 1, 2, 3),
            mlp_ratio=4,
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.05),
        fuser=dict(
            type='NATFuserBlock',
            kernel_size=5,
            gated=True,
            post_attn=True,
            attn_mode='cross')),
    preprocessor_sec=[
        [
            'dct',
            dict(
                type='DCTProcessor',
                in_channels=1,
                embed_dims=32,
                num_heads=1,
                patch_size=3,
                stride=1,
                sr_ratio=4,
                out_channels=32,
                norm_cfg=norm_cfg,
                reduce_neg=False)
        ],
        ['ela', dict(type='NoFilter')],
        ['img', dict(type='SRMConv2d_simple', inc=3, learnable=False)],
    ],
    decode_head=dict(
        type='LitePrototypeHead',
        in_channels=[32, 64, 160, 256],
        in_index=[0, 1, 2, 3],
        channels=128,
        dropout_ratio=0.1,
        num_classes=num_classes,
        norm_cfg=norm_cfg,
        align_corners=False,
        up_decode=True,
        dim=64,
        temperature=0.1,
        momentum=0.99,
        loss_weight_contrastive=0.05,
        sep_margin=0.0,
        loss_weight_sep=0.01,
        proto_fuse_weight=0.5,
        loss_decode=[
            dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            dict(type='DiceLoss', use_sigmoid=False, loss_weight=0.5, eps=1e-5, ignore_index=255),
        ]),
    train_cfg=dict(),
    # Exact 8x8 DCT & 32-divisor aligned Full-Scan sliding window (no Early Exit)
    test_cfg=dict(mode='slide', crop_size=(512, 512), stride=(384, 384)),
)

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

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='ELA', quality=quality),
    dict(type='BlockDCT', zigzag=True),
    dict(type='LoadAnnotations', binary=True),
    dict(type='PackSegInputsWithExtra', extra_keys=('dct', 'ela')),
]

train_dataloader = dict(
    batch_size=16,
    num_workers=8,
    persistent_workers=True,
    pin_memory=True,
    dataset=dict(pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    dataset=dict(pipeline=test_pipeline))

test_dataloader = val_dataloader

optim_wrapper = dict(
    _delete_=True,
    type='AmpOptimWrapper',
    optimizer=dict(
        type='AdamW', lr=1.2e-4, betas=(0.9, 0.999), weight_decay=0.01),
    clip_grad=dict(max_norm=1.0),
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.),
            'rpb': dict(decay_mult=0.),
        }),
    loss_scale='dynamic')

# 60,000 iters x batch_size 16 = 960,000 training crops (identical to ASC-Former PR 2025 paper)
train_cfg = dict(type='IterBasedTrainLoop', max_iters=60000, val_interval=5000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-6,
        by_epoch=False,
        begin=0,
        end=2000),
    dict(
        type='CosineAnnealingLR',
        eta_min=1e-6,
        by_epoch=False,
        begin=2000,
        end=60000),
]

val_evaluator = dict(
    type='BinaryIoUMetric',
    iou_metrics=['mIoU', 'mDice', 'mFscore'],
    output_dir='work_dirs/b0_lite_kd_960k/val_metrics',
    save_confusion_matrix=True)
test_evaluator = val_evaluator

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=100, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=5000,
        max_keep_ckpts=2,
        save_best='mIoU',
        rule='greater'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'),
    save_result=dict(type='SegResultHook', interval=1, binary=True))

find_unused_parameters = True
