"""A100-Turbo Configuration: Batch Size = 8, 20,000 Iters (~3.5 Hours, ~20GB VRAM).

Thiết kế tối ưu bứt tốc cho NVIDIA A100 (40GB/80GB):
1. Batch Size = 8: Tận dụng ~19-20GB VRAM (gấp đôi bản cũ), khai thác tối đa song song Tensor Cores.
2. 20,000 iterations: Học trọn vẹn 160,000 mẫu ảnh (chính xác bằng 40,000 iters của bản batch 4).
3. Thời gian rút ngắn 50%: Từ ~7 tiếng xuống chỉ còn ~3.2 - 3.5 tiếng!
4. Tỉ lệ mẫu dương tính pos_prob = 0.75: Đảm bảo trung bình 6/8 ảnh trong batch chứa nét chữ giả.
5. Base LR = 8e-5 (scale theo Linear Scaling Rule cho batch 8), Decode Head nhận 8e-4.
6. Validate mỗi 5,000 iters, tự động lưu best_mIoU.
"""
_base_ = ['./ascformer_rtm_b0_tristream_proto_40k.py']

crop_size = (512, 512)
quality = 80
pos_prob = 0.75

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

# Tối ưu DataLoader cho A100 (Batch Size = 8, 8 Workers, Pin Memory DMA)
train_dataloader = dict(
    batch_size=8,
    num_workers=8,
    persistent_workers=True,
    pin_memory=True,
    dataset=dict(pipeline=train_pipeline)
)

val_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    pin_memory=True
)

test_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    pin_memory=True,
    dataset=dict(ann_file='val.txt')
)

# 20,000 iterations x 8 crops = 160,000 crops (~27.6 epochs), validate mỗi 5,000 bước
train_cfg = dict(max_iters=20000, val_interval=5000)

# Warmup 1000 bước + PolyLR decay mượt mà đến 20000 bước
param_scheduler = [
    dict(type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1000),
    dict(type='PolyLR', eta_min=0., power=1., by_epoch=False, begin=1000, end=20000),
]

# OptimWrapper với lr=8e-5 cho batch_size=8, Decode Head lr_mult=10.0 (nhận 8e-4)
optim_wrapper = dict(
    type='AmpOptimWrapper',
    loss_scale='dynamic',
    optimizer=dict(
        type='AdamW',
        lr=0.00008,
        betas=(0.9, 0.999),
        weight_decay=0.01
    ),
    paramwise_cfg=dict(
        custom_keys=dict(
            pos_block=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0),
            head=dict(lr_mult=10.0),
            rpb=dict(decay_mult=0.0)
        )
    )
)

# Lưu checkpoint mỗi 5,000 iters, theo dõi và lưu checkpoint có mIoU cao nhất
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=5000,
        max_keep_ckpts=4,
        save_best='mIoU',
        rule='greater'
    )
)

val_evaluator = dict(
    type='BinaryIoUMetric',
    iou_metrics=['mIoU', 'mDice', 'mFscore'],
    threshold=0.5,
    output_dir='work_dirs/tristream_proto_a100_turbo/val_metrics',
    save_confusion_matrix=True
)

test_evaluator = dict(
    type='BinaryIoUMetric',
    iou_metrics=['mIoU', 'mDice', 'mFscore'],
    threshold=0.25,
    output_dir='work_dirs/tristream_proto_a100_turbo/test_metrics',
    save_confusion_matrix=True
)

# Khắc phục triệt để xung đột TensorFlow/JAX/NumPy trên Colab
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='SegLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer'
)
