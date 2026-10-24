"""Single-GPU DWT pilot with the same training budget as the baseline."""
_base_ = ['./ascformer_rtm_dwt.py']
randomness = dict(seed=3407)
train_dataloader = dict(batch_size=1, num_workers=2, persistent_workers=True)
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
    output_dir='work_dirs/pilot8k_dwt/val_metrics',
    save_confusion_matrix=True)

