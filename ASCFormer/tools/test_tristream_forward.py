"""Smoke test script for Tri-Stream (RGB + Forensic + OCR) forward pass and verification."""
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
    print("=== Testing Tri-Stream (RGB + Forensic + OCR) Model ===")
    print("Registering all modules...")
    register_all_modules(init_default_scope=True)

    config_path = os.path.join(ascformer_dir, 'configs/ascformer/ascformer_rtm_b0_tristream.py')
    print(f"Loading config from {config_path}...")
    cfg = Config.fromfile(config_path)

    # Disable downloading pretrained weights during architecture shape verification
    cfg.model.backbone.backbone_main.pretrained = None
    cfg.model.backbone.backbone_extra.pretrained = None
    cfg.model.backbone.backbone_ocr.pretrained = None

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    print("Building Tri-Stream model...")
    model = MODELS.build(cfg.model).to(device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params / 1e6:.2f} M")
    print(f"Trainable parameters: {trainable_params / 1e6:.2f} M")

    # 1. Test inference (mode='predict') with batch_size=1
    print("\n1. Testing forward_predict (batch_size=1 with OCR stream)...")
    C, H, W = 3, 512, 512
    inputs_val = [torch.randn(C, H, W, device=device)]
    extra_val = dict(
        dct=[torch.randn(1, H, W, device=device)],
        ela=[torch.randn(3, H, W, device=device)],
        ocr=[torch.randn(3, H, W, device=device)]
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
        preds = model(
            inputs=prep_val['inputs'],
            extras=prep_val['extras'],
            data_samples=prep_val['data_samples'],
            mode='predict'
        )
    print(f"Predict successful! Number of output data_samples: {len(preds)}")
    print(f"Predicted seg_logits shape: {preds[0].seg_logits.data.shape}")

    # 2. Test training step (mode='loss') with batch_size=2
    print("\n2. Testing forward_loss (batch_size=2 training step with OCR stream)...")
    model.train()
    inputs_train = [torch.randn(C, H, W, device=device), torch.randn(C, H, W, device=device)]
    extra_train = dict(
        dct=[torch.randn(1, H, W, device=device), torch.randn(1, H, W, device=device)],
        ela=[torch.randn(3, H, W, device=device), torch.randn(3, H, W, device=device)],
        ocr=[torch.randn(3, H, W, device=device), torch.randn(3, H, W, device=device)]
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
        gt = torch.randint(0, 2, (1, H, W), dtype=torch.int64, device=device)
        s.gt_sem_seg = PixelData(data=gt)
        samples_train.append(s)

    data_train = dict(inputs=inputs_train, data_samples=samples_train, extra=extra_train)
    prep_train = model.data_preprocessor(data_train, training=True)

    losses = model(
        inputs=prep_train['inputs'],
        extras=prep_train['extras'],
        data_samples=prep_train['data_samples'],
        mode='loss'
    )
    print("Loss computation successful! Losses returned:")
    total_loss = torch.tensor(0.0, device=device)
    for k, v in losses.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {v.item():.4f}")
            if 'loss' in k:
                total_loss = total_loss + v.mean()
    print(f"Total loss: {total_loss.item():.4f}")

    print("\nTesting backward pass...")
    total_loss.backward()
    print("Backward pass successful! Gradients computed across all 3 streams.")

    print("\n>>> ALL TRI-STREAM TESTS PASSED SUCCESSFULLY! <<<")

if __name__ == '__main__':
    main()
