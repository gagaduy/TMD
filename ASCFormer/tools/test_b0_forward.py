"""Smoke test script for MiT-B0 baseline model building and forward pass."""
import os
import sys
import torch

# Ensure ASCFormer is in python path
current_dir = os.path.dirname(os.path.abspath(__file__))
ascformer_dir = os.path.abspath(os.path.join(current_dir, '..'))
if ascformer_dir not in sys.path:
    sys.path.insert(0, ascformer_dir)

from mmengine.config import Config
from mmseg.registry import MODELS
from mmseg.utils import register_all_modules
from mmseg.structures import SegDataSample
from mmengine.structures import PixelData

def main():
    print("Registering all modules...")
    register_all_modules(init_default_scope=True)

    config_path = os.path.join(ascformer_dir, 'configs/ascformer/ascformer_rtm_b0.py')
    print(f"Loading config from {config_path}...")
    cfg = Config.fromfile(config_path)

    # Disable downloading pretrained weights during architecture shape verification
    cfg.model.backbone.backbone_main.pretrained = None
    cfg.model.backbone.backbone_extra.pretrained = None

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    print("Building model...")
    model = MODELS.build(cfg.model).to(device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params / 1e6:.2f} M")
    print(f"Trainable parameters: {trainable_params / 1e6:.2f} M")

    # 1. Test inference (mode='predict') with batch_size=1
    print("Testing forward_predict (batch_size=1)...")
    C, H, W = 3, 512, 512
    inputs_val = [torch.randn(C, H, W, device=device)]
    extra_val = dict(
        dct=[torch.randn(1, H, W, device=device)],
        ela=[torch.randn(3, H, W, device=device)]
    )
    sample_val = SegDataSample()
    sample_val.set_metainfo(dict(
        img_shape=(H, W),
        ori_shape=(H, W),
        pad_shape=(H, W),
        padding_size=[0, 0, 0, 0]
    ))
    data_val = dict(inputs=inputs_val, data_samples=[sample_val], extra=extra_val)

    with torch.no_grad():
        prep_val = model.data_preprocessor(data_val, training=False)
        preds = model(prep_val['inputs'], prep_val['extras'], prep_val['data_samples'], mode='predict')
        print(f"Predict successful! Number of output samples: {len(preds)}")
        print(f"Pred shape: {preds[0].pred_sem_seg.data.shape}")

    # 2. Test training loss (mode='loss') with batch_size=2
    print("Testing forward_loss (batch_size=2)...")
    model.train()
    inputs_train = [torch.randn(C, H, W, device=device) for _ in range(2)]
    extra_train = dict(
        dct=[torch.randn(1, H, W, device=device) for _ in range(2)],
        ela=[torch.randn(3, H, W, device=device) for _ in range(2)]
    )
    samples_train = []
    for _ in range(2):
        s = SegDataSample()
        s.set_metainfo(dict(
            img_shape=(H, W),
            ori_shape=(H, W),
            pad_shape=(H, W),
            padding_size=[0, 0, 0, 0]
        ))
        s.gt_sem_seg = PixelData(data=torch.zeros((1, H, W), dtype=torch.long, device=device))
        samples_train.append(s)

    data_train = dict(inputs=inputs_train, data_samples=samples_train, extra=extra_train)
    train_data = model.data_preprocessor(data_train, training=True)
    losses = model(train_data['inputs'], train_data['extras'], train_data['data_samples'], mode='loss')
    print("Losses calculated successfully:")
    for k, v in losses.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {v.item():.4f}")
        elif isinstance(v, list):
            print(f"  {k}: {[x.item() for x in v]}")

    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 2)
        reserved = torch.cuda.memory_reserved() / (1024 ** 2)
        print(f"CUDA memory: allocated={allocated:.1f} MB, reserved={reserved:.1f} MB")

    print("\n>>> ALL SMOKE TESTS PASSED FOR MiT-B0! <<<")

if __name__ == '__main__':
    main()
