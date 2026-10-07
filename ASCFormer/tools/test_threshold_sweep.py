"""Comprehensive Threshold Sweep Evaluation Tool for Real Text Manipulation (RTM).

Computes Tampered IoU, Precision, Recall, Dice, and mIoU across multiple decision
thresholds (e.g. tau in [0.10, 0.50]) in a SINGLE inference pass.
"""

import argparse
import os
import sys
import torch
import numpy as np
from prettytable import PrettyTable
from mmengine.config import Config
from mmengine.runner import Runner
from mmseg.utils import register_all_modules

# Ensure safe loading with torch 2.6+
_orig_load = torch.load
def _safe_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _orig_load(*args, **kwargs)
torch.load = _safe_load


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate RTM model across multiple decision thresholds')
    parser.add_argument('config', help='Path to model config file')
    parser.add_argument('checkpoint', help='Path to checkpoint file (.pth)')
    parser.add_argument(
        '--thresholds',
        nargs='+',
        type=float,
        default=[0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50],
        help='List of decision thresholds to evaluate')
    parser.add_argument(
        '--ann-file',
        type=str,
        default='val.txt',
        help='Annotation file to evaluate on (val.txt or test.txt)')
    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='Max number of images to evaluate (for fast preview)')
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1,
        help='Inference batch size (must be 1 for sliding-window)')
    parser.add_argument(
        '--work-dir',
        type=str,
        default=None,
        help='Directory to save sweep results JSON')
    return parser.parse_args()


def main():
    args = parse_args()
    register_all_modules()

    # Load configuration
    cfg = Config.fromfile(args.config)
    cfg.load_from = args.checkpoint
    if args.work_dir:
        cfg.work_dir = args.work_dir
    else:
        cfg.work_dir = './work_dirs/threshold_sweep'

    # Configure validation / test dataloader to use specified ann_file
    cfg.test_dataloader.dataset.ann_file = args.ann_file
    cfg.test_dataloader.batch_size = args.batch_size

    # Build runner
    runner = Runner.from_cfg(cfg)
    model = runner.model
    model.eval()

    test_dataloader = runner.test_dataloader
    thresholds = sorted(args.thresholds)

    # Accumulators for each threshold: [TP, FP, FN, TN]
    confusion = {t: {'TP': 0, 'FP': 0, 'FN': 0, 'TN': 0} for t in thresholds}
    total_samples = len(test_dataloader.dataset)
    eval_samples = min(total_samples, args.max_samples) if args.max_samples else total_samples

    print('=' * 85)
    print('🔬 RTM THRESHOLD SWEEP EVALUATION')
    print(f'   Model Config : {args.config}')
    print(f'   Checkpoint   : {args.checkpoint}')
    print(f'   Dataset Split: {args.ann_file} ({eval_samples} images)')
    print(f'   Thresholds   : {thresholds}')
    print('=' * 85)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    count = 0

    with torch.no_grad():
        for i, data_batch in enumerate(test_dataloader):
            if args.max_samples and i >= args.max_samples:
                break

            # Forward pass using runner / model test_step
            outputs = model.test_step(data_batch)

            for output in outputs:
                count += 1
                gt_data = output.gt_sem_seg.data.squeeze().to(device)
                ignore_mask = (gt_data == 255)

                # Get predicted logits
                seg_logits = output.seg_logits.data.to(device)
                if seg_logits.shape[0] == 2:
                    # Softmax probability of tampered class (channel 1)
                    prob_tamp = torch.softmax(seg_logits.float(), dim=0)[1]
                else:
                    prob_tamp = torch.sigmoid(seg_logits.float()).squeeze()

                gt_bin = (gt_data == 1) & (~ignore_mask)
                gt_auth = (gt_data == 0) & (~ignore_mask)

                # Vectorized confusion update for all thresholds
                for t in thresholds:
                    pred_tamp = (prob_tamp > t) & (~ignore_mask)

                    tp = (pred_tamp & gt_bin).sum().item()
                    fp = (pred_tamp & gt_auth).sum().item()
                    fn = ((~pred_tamp) & gt_bin).sum().item()
                    tn = ((~pred_tamp) & gt_auth).sum().item()

                    confusion[t]['TP'] += tp
                    confusion[t]['FP'] += fp
                    confusion[t]['FN'] += fn
                    confusion[t]['TN'] += tn

            if (i + 1) % 50 == 0 or (i + 1) == eval_samples:
                print(f'   Đang xử lý: [{count}/{eval_samples}] ảnh...')

    # Compile table
    table = PrettyTable()
    table.field_names = [
        'Threshold (tau)',
        'Tampered IoU (%)',
        'Precision (%)',
        'Recall (%)',
        'Dice / F1 (%)',
        'Background IoU (%)',
        'mIoU (%)'
    ]

    best_iou = -1.0
    best_tau = None
    results_summary = []

    for t in thresholds:
        tp = confusion[t]['TP']
        fp = confusion[t]['FP']
        fn = confusion[t]['FN']
        tn = confusion[t]['TN']

        # Tampered metrics
        t_iou = (tp / (tp + fp + fn + 1e-8)) * 100
        t_prec = (tp / (tp + fp + 1e-8)) * 100
        t_rec = (tp / (tp + fn + 1e-8)) * 100
        t_dice = (2.0 * tp / (2.0 * tp + fp + fn + 1e-8)) * 100

        # Background metrics
        bg_iou = (tn / (tn + fp + fn + 1e-8)) * 100

        # Mean IoU
        m_iou = 0.5 * (bg_iou + t_iou)

        if t_iou > best_iou:
            best_iou = t_iou
            best_tau = t

        row = [
            f'{t:.2f}',
            f'{t_iou:.2f}%',
            f'{t_prec:.2f}%',
            f'{t_rec:.2f}%',
            f'{t_dice:.2f}%',
            f'{bg_iou:.2f}%',
            f'{m_iou:.2f}%'
        ]
        table.add_row(row)
        results_summary.append({
            'threshold': t,
            'tampered_iou': round(t_iou, 4),
            'tampered_precision': round(t_prec, 4),
            'tampered_recall': round(t_rec, 4),
            'tampered_dice': round(t_dice, 4),
            'bg_iou': round(bg_iou, 4),
            'm_iou': round(m_iou, 4)
        })

    print('\n' + '=' * 85)
    print('📊 BẢNG KẾT QUẢ QUÉT NGƯỠNG TOÀN DIỆN (THRESHOLD SWEEP)')
    print('=' * 85)
    print(table)
    print(f'\n⭐ NGƯỠNG ĐẠT ĐỈNH (PEAK): tau = {best_tau:.2f} với Tampered IoU = {best_iou:.2f}%')
    print('=' * 85)


if __name__ == '__main__':
    main()
