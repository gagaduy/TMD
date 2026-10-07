"""Head memorization diagnosis on cached frozen-backbone features, CPU FP32."""
import copy
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
from mmengine.config import Config
from mmengine.dataset import pseudo_collate
from mmengine.optim import build_optim_wrapper
from mmseg.registry import DATASETS, MODELS
from mmseg.utils import register_all_modules

torch.set_num_threads(4)
random.seed(3407)
np.random.seed(3407)
torch.manual_seed(3407)
register_all_modules()
outdir = Path('work_dirs/baseline8k_overfit_head')
outdir.mkdir(exist_ok=True)
cfg = Config.fromfile('configs/ascformer/ascformer_rtm.py')
cfg.model.backbone.backbone_main.pretrained = None
cfg.model.backbone.backbone_extra.pretrained = None
dataset = DATASETS.build(cfg.train_dataloader.dataset)
samples = []
for i in range(min(len(dataset), 100)):
    item = dataset[i]
    if (item['data_samples'].gt_sem_seg.data == 1).sum().item() >= 1000:
        h, w = item['inputs'].shape[-2:]
        item['data_samples'].set_metainfo(dict(
            ori_shape=(h, w), img_shape=(h, w), pad_shape=(h, w),
            flip=False, padding_size=[0, 0, 0, 0]))
        samples.append(item)
        print('SELECT', item['data_samples'].img_path, flush=True)
    if len(samples) == 3:
        break
assert len(samples) == 3
model = MODELS.build(cfg.model).cpu()
saved = torch.load('work_dirs/pilot8k_baseline/iter_8000.pth',
                   map_location='cpu', weights_only=False)
model.load_state_dict(saved['state_dict'], strict=True)
del saved
model.eval()
for name, parameter in model.named_parameters():
    parameter.requires_grad_(name.startswith('decode_head.'))
cached = []
with torch.no_grad():
    for item in samples:
        data = model.data_preprocessor(pseudo_collate([copy.deepcopy(item)]), training=True)
        features = model.forward_encoder(data['inputs'], data['extras'])
        cached.append((features, data['data_samples']))
optimizer = build_optim_wrapper(model, cfg.optim_wrapper)
history = []

def evaluate(step):
    model.eval()
    rows = []
    with torch.no_grad():
        for features, data_samples in cached:
            logits = model.decode_head.predict(features, [s.metainfo for s in data_samples], model.test_cfg)
            pred = model.postprocess_result(logits, copy.deepcopy(data_samples))[0]
            p = pred.pred_sem_seg.data.squeeze() == 1
            gt = pred.gt_sem_seg.data.squeeze() == 1
            tp = (p & gt).sum().item()
            fp = (p & ~gt).sum().item()
            fn = (~p & gt).sum().item()
            rows.append(dict(TP=tp, FP=fp, FN=fn,
                             IoU=tp/max(tp+fp+fn, 1),
                             Dice=2*tp/max(2*tp+fp+fn, 1)))
    row = dict(step=step, crops=rows)
    history.append(row)
    print('EVAL', json.dumps(row), flush=True)
    (outdir/'results.json').write_text(json.dumps(history, indent=2))
    model.decode_head.train()
    return min(r['IoU'] for r in rows)

evaluate(0)
started = time.perf_counter()
for step in range(1, 301):
    features, data_samples = cached[(step-1) % len(cached)]
    losses = {'decode.' + key: value for key, value in
              model.decode_head.loss(features, data_samples, model.train_cfg).items()}
    loss, log_vars = model.parse_losses(losses)
    assert torch.isfinite(loss), log_vars
    optimizer.zero_grad()
    loss.backward()
    grad = model.decode_head.conv_seg.weight.grad
    assert grad is not None and torch.isfinite(grad).all()
    if step == 1:
        before = model.decode_head.conv_seg.weight.detach().clone()
    optimizer.step()
    if step == 1:
        print('FIRST_UPDATE', (model.decode_head.conv_seg.weight-before).abs().max().item(), flush=True)
    if step == 1 or step % 10 == 0:
        print('TRAIN', step, 'loss', loss.item(), 'ce', losses['decode.loss_ce'].item(),
              'head_grad_norm', grad.norm().item(),
              'elapsed', time.perf_counter()-started, flush=True)
    if step % 50 == 0:
        min_iou = evaluate(step)
        if min_iou >= .8:
            break
torch.save(dict(state_dict=model.state_dict(), diagnostic_only=True, steps=step),
           outdir/'memorization.pth')
print('DONE', step, flush=True)
