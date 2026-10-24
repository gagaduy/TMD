"""Read-only checkpoint diagnosis on fixed positive training crops, on CPU."""
import copy
import json
import random
import sys
from pathlib import Path
import numpy as np
import torch
from mmengine.config import Config
from mmengine.dataset import pseudo_collate
from mmseg.registry import DATASETS, MODELS
from mmseg.utils import register_all_modules

torch.set_num_threads(4)
random.seed(3407)
np.random.seed(3407)
torch.manual_seed(3407)
register_all_modules()
cfg = Config.fromfile('configs/ascformer/ascformer_rtm_pilot8k.py')
cfg.model.backbone.backbone_main.pretrained = None
cfg.model.backbone.backbone_extra.pretrained = None
dataset = DATASETS.build(cfg.train_dataloader.dataset)
samples = []
for i in range(min(len(dataset), 100)):
    item = dataset[i]
    gt = item['data_samples'].gt_sem_seg.data
    if (gt == 1).sum().item() >= 1000:
        samples.append(item)
        print('SELECT', i, item['data_samples'].img_path,
              'positive_pixels', (gt == 1).sum().item(), flush=True)
    if len(samples) == 3:
        break
assert len(samples) == 3
model = MODELS.build(cfg.model).cpu().eval()
report = []
cases = [
    ('baseline8k', 'work_dirs/pilot8k_baseline/iter_8000.pth'),
    ('paper', 'work_dirs/ascformer_rtm/ascformer_model.pth'),
]
if '--bn-only' in sys.argv:
    cases = [('baseline8k_batchstats', 'work_dirs/pilot8k_baseline/iter_8000.pth')]
for name, checkpoint in cases:
    # These are the local training output and the previously verified paper file.
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    model.load_state_dict(saved['state_dict'], strict=True)
    del saved
    model.eval()
    if name.endswith('batchstats'):
        # Diagnostic only: use current-crop statistics without modifying saved
        # weights or enabling dropout. Restore original buffers after each crop.
        bn_buffers = {k: v.clone() for k, v in model.named_buffers()}
        for module in model.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.train()
    for i, sample in enumerate(samples):
        batch = pseudo_collate([copy.deepcopy(sample)])
        # Cropped input needs cropped output metadata, not the source image size.
        h, w = batch['inputs'][0].shape[-2:]
        batch['data_samples'][0].set_metainfo(dict(
            ori_shape=(h, w), img_shape=(h, w), pad_shape=(h, w),
            flip=False, padding_size=[0, 0, 0, 0]))
        with torch.inference_mode():
            out = model.test_step(batch)[0]
        logits = out.seg_logits.data
        prob = logits.softmax(dim=0)[1]
        pred = out.pred_sem_seg.data.squeeze() == 1
        gt = out.gt_sem_seg.data.squeeze() == 1
        tp = (pred & gt).sum().item()
        fp = (pred & ~gt).sum().item()
        fn = (~pred & gt).sum().item()
        row = dict(model=name, crop=i, image=sample['data_samples'].img_path,
                   gt_pixels=gt.sum().item(), predicted_pixels=pred.sum().item(),
                   TP=tp, FP=fp, FN=fn, IoU=tp / max(tp+fp+fn, 1),
                   probability_min=prob.min().item(), probability_max=prob.max().item(),
                   probability_on_gt=prob[gt].mean().item(),
                   probability_on_background=prob[~gt].mean().item(),
                   finite=bool(torch.isfinite(logits).all()))
        report.append(row)
        print('RESULT', json.dumps(row), flush=True)
        if name.endswith('batchstats'):
            for key, value in model.named_buffers():
                value.copy_(bn_buffers[key])
Path('work_dirs/baseline8k_probe').mkdir(exist_ok=True)
filename = 'bn_results.json' if '--bn-only' in sys.argv else 'results.json'
Path('work_dirs/baseline8k_probe', filename).write_text(json.dumps(report, indent=2))
