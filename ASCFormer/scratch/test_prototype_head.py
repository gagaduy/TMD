import torch
from mmseg.utils import register_all_modules
from mmseg.registry import MODELS
from mmseg.structures import SegDataSample
from mmengine.structures import PixelData

register_all_modules()

def test_prototype_head():
    print("Testing PrototypeContrastiveHead...")
    head_cfg = dict(
        type='PrototypeContrastiveHead',
        in_channels=[32, 64, 160, 256],
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=False,
        up_decode=True,
        dim=128,
        temperature=0.1,
        momentum=0.99,
        loss_weight_contrastive=0.05,
        sep_margin=0.0,
        loss_weight_sep=0.01,
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)
    )
    
    head = MODELS.build(head_cfg).cuda()
    
    # 4 stage features
    B = 2
    inputs = [
        torch.randn(B, 32, 128, 128, device='cuda', requires_grad=True),
        torch.randn(B, 64, 64, 64, device='cuda', requires_grad=True),
        torch.randn(B, 160, 32, 32, device='cuda', requires_grad=True),
        torch.randn(B, 256, 16, 16, device='cuda', requires_grad=True)
    ]
    
    # Create batch data samples with ground truth
    data_samples = []
    for b in range(B):
        sample = SegDataSample()
        # 512x512 mask with some positive pixels
        gt = torch.zeros(1, 512, 512, dtype=torch.long, device='cuda')
        gt[:, 100:150, 100:200] = 1 # tampered region
        sample.gt_sem_seg = PixelData(data=gt)
        sample.set_metainfo(dict(ori_shape=(512, 512), img_shape=(512, 512), pad_shape=(512, 512)))
        data_samples.append(sample)
        
    # Forward and loss
    losses = head.loss(inputs, data_samples, train_cfg=dict())
    print("Loss outputs:")
    for k, v in losses.items():
        print(f"  {k}: {v.item():.4f}")
        
    assert 'loss_ce' in losses
    assert 'loss_contrastive' in losses
    
    total_loss = sum(losses.values())
    total_loss.backward()
    print("Backward pass succeeded!")
    
    # Verify gradients
    for i, inp in enumerate(inputs):
        assert inp.grad is not None, f"Input {i} has no gradient!"
        print(f"  Input {i} grad norm: {inp.grad.norm().item():.4f}")
        
    # Verify prototype updates
    print("Prototypes initialized:", head.proto_initialized.tolist())
    print("Prototype 0 norm:", head.prototypes[0].norm().item())
    print("Prototype 1 norm:", head.prototypes[1].norm().item())
    assert head.proto_initialized.all()
    
    # Test predict
    seg_logits = head.predict(inputs, [d.metainfo for d in data_samples], test_cfg=dict())
    print("Predict output shape:", seg_logits.shape)
    assert seg_logits.shape == (B, 2, 512, 512)
    
    print("\nALL TESTS PASSED FOR PrototypeContrastiveHead!")

if __name__ == '__main__':
    test_prototype_head()
