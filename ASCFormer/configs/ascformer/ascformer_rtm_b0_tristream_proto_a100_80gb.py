"""High-Capacity Tri-Stream MiT-B0 pilot for NVIDIA A100 80GB (< 1 hour budget).

Tận dụng sức mạnh của NVIDIA A100 80GB VRAM:
- Batch size = 4 (hoặc 6), num_workers = 8
- 8,000 iterations chuẩn (đủ số bước gradient updates cho ViT + Dual Prototypes hội tụ)
- Học 32,000 mẫu ảnh (gấp đôi bản pilot cũ)
- Base LR = 6e-5 chuẩn xác, triệt tiêu sốc loss ở Decode Head
- Hoàn thành trong ~50-55 phút (< 1 giờ)
"""
_base_ = ['./ascformer_rtm_b0_tristream_proto_pilot8k.py']

# Tối ưu DataLoader cho A100-SXM4 (80GB VRAM + high vCPU)
train_dataloader = dict(
    batch_size=4,
    num_workers=8,
    persistent_workers=True
)

val_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True
)

# 8,000 iterations chuẩn (~48-50 phút train trên A100)
train_cfg = dict(max_iters=8000, val_interval=8000)

# Warmup 800 bước + PolyLR decay mượt mà
param_scheduler = [
    dict(type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=800),
    dict(type='PolyLR', eta_min=0., power=1., by_epoch=False, begin=800, end=8000),
]

# Base LR 6e-5 chuẩn để Decode Head (lr_mult=10) nhận 6e-4 an toàn
optim_wrapper = dict(
    type='AmpOptimWrapper',
    loss_scale='dynamic',
    optimizer=dict(
        type='AdamW',
        lr=0.00006,
        betas=(0.9, 0.999),
        weight_decay=0.01
    )
)

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=8000,
        max_keep_ckpts=1,
        save_best=None
    )
)

val_evaluator = dict(
    iou_metrics=['mIoU', 'mDice', 'mFscore'],
    output_dir='work_dirs/a100_80gb_tristream_proto/val_metrics',
    save_confusion_matrix=True
)

test_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    dataset=dict(ann_file='val.txt')
)

test_evaluator = dict(
    type='BinaryIoUMetric',
    iou_metrics=['mIoU', 'mDice', 'mFscore'],
    threshold=0.25,
    output_dir='work_dirs/a100_80gb_tristream_proto/test_metrics',
    save_confusion_matrix=True
)

# Use LocalVisBackend only (prevents TensorFlow/JAX/NumPy conflict on Colab)
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='SegLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer'
)

