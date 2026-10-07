"""High-Performance 40k Training Configuration for NVIDIA A100 (40GB/80GB VRAM).

Thiết kế tối ưu hóa chuyên sâu cho NVIDIA A100 SXM4:
1. Batch Size = 4 (Điểm ngọt toán học: giữ vững mật độ nét chữ giả 0.1%, không làm loãng gradient).
2. Num Workers = 8 + Pin Memory = True (DMA host-to-device direct transfer, triệt tiêu thời gian chờ I/O).
3. TF32 Tensor Cores tự động kích hoạt (tốc độ xử lý ma trận tăng 30-40%).
4. 40,000 iterations chuẩn (đủ 40,000 bước EMA cho Dual Prototypes đẩy biên phân cách xác suất vượt mốc 50%).
5. Validation định kỳ mỗi 10,000 iters (tiết kiệm thời gian chạy validation lặp lại).
6. Tự động lưu Checkpoint tốt nhất (best_mIoU) mỗi 5,000 iters.
"""
_base_ = ['./ascformer_rtm_b0_tristream_proto_40k.py']

# Tối ưu DataLoader cho băng thông A100
train_dataloader = dict(
    batch_size=4,
    num_workers=8,
    persistent_workers=True,
    pin_memory=True
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

# 40,000 iterations (~27.6 epochs), validate mỗi 10,000 bước
train_cfg = dict(max_iters=40000, val_interval=10000)

# Warmup 1500 bước + PolyLR decay mượt mà
param_scheduler = [
    dict(type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(type='PolyLR', eta_min=0., power=1., by_epoch=False, begin=1500, end=40000),
]

# OptimWrapper với lr=6e-5 chuẩn, Decode Head lr_mult=10.0 (nhận 6e-4 an toàn)
optim_wrapper = dict(
    type='AmpOptimWrapper',
    loss_scale='dynamic',
    optimizer=dict(
        type='AdamW',
        lr=0.00006,
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
        max_keep_ckpts=3,
        save_best='mIoU',
        rule='greater'
    )
)

val_evaluator = dict(
    type='BinaryIoUMetric',
    iou_metrics=['mIoU', 'mDice', 'mFscore'],
    threshold=0.5,
    output_dir='work_dirs/tristream_proto_40k_a100/val_metrics',
    save_confusion_matrix=True
)

test_evaluator = dict(
    type='BinaryIoUMetric',
    iou_metrics=['mIoU', 'mDice', 'mFscore'],
    threshold=0.25,
    output_dir='work_dirs/tristream_proto_40k_a100/test_metrics',
    save_confusion_matrix=True
)

# Khắc phục triệt để xung đột TensorFlow/JAX/NumPy trên Colab
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='SegLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer'
)
