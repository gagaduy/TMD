"""SOTA Mobile MiT-B0 30,000 Iterations Training Configuration for Google Colab / A100.

Mô hình siêu nhẹ dành cho di động (Mobile Super-Fast):
1. Backbone: SegFormer MiT-B0 siêu nhẹ (~3.7M params/stream, 1/6 so với MiT-B2).
2. Tích hợp đầy đủ cả 3 đóng góp:
   - Tri-Stream Architecture (RGB + ELA Tần số + OCR Stream)
   - Dual-Prototype Contrastive Head (Lực kéo nguyên mẫu phân lớp)
   - Focused Tamper Cropping (Tập trung mẫu can thiệp)
3. Quy mô huấn luyện chuẩn:
   - 30,000 iterations x batch_size 4 (hoặc 8)
   - AmpOptimWrapper Mixed Precision FP16/TF32
   - Lưu checkpoint mỗi 5,000 bước, validate mỗi 10,000 bước
"""
_base_ = ['./ascformer_rtm_b0_ablation16k_m4_proto.py']

# Tối ưu DataLoader cho Colab (GPU T4 / V100 / A100 / L4)
train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    pin_memory=True
)

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    pin_memory=True
)

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    dataset=dict(ann_file='val.txt')
)

# 30,000 iterations chuẩn, validate mỗi 10,000 bước
train_cfg = dict(max_iters=30000, val_interval=10000)

# Warmup 1500 bước + PolyLR decay
param_scheduler = [
    dict(type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(type='PolyLR', eta_min=0., power=1., by_epoch=False, begin=1500, end=30000),
]

# AmpOptimWrapper hỗn hợp độ chính xác cực nhanh
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
    output_dir='work_dirs/b0_30k_sota_mobile/val_metrics',
    save_confusion_matrix=True
)
