"""Create a deterministic, prefix-stratified validation split for RTM."""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def _group_name(sample_id):
    return sample_id.split('_', 1)[0]


def _read_sample_ids(source):
    sample_ids = [line.strip() for line in source.read_text().splitlines()]
    sample_ids = [sample_id for sample_id in sample_ids if sample_id]
    if not sample_ids:
        raise ValueError(f'No sample IDs found in {source}')
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError(f'Duplicate sample IDs found in {source}')
    return sample_ids


def create_split(source, output_dir, val_ratio=0.1, seed=20260903):
    """Write train.txt and val.txt while preserving each edit-type ratio."""
    if not 0 < val_ratio < 1:
        raise ValueError('val_ratio must be between 0 and 1')

    source = Path(source)
    output_dir = Path(output_dir)
    groups = defaultdict(list)
    for sample_id in _read_sample_ids(source):
        groups[_group_name(sample_id)].append(sample_id)

    rng = random.Random(seed)
    train_ids = []
    val_ids = []
    group_counts = {}
    for group in sorted(groups):
        group_ids = sorted(groups[group])
        rng.shuffle(group_ids)
        val_count = max(1, round(len(group_ids) * val_ratio))
        val_count = min(val_count, len(group_ids) - 1)
        group_val = group_ids[:val_count]
        group_train = group_ids[val_count:]
        train_ids.extend(group_train)
        val_ids.extend(group_val)
        group_counts[group] = dict(train=len(group_train), val=len(group_val))

    train_ids.sort()
    val_ids.sort()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'train.txt').write_text('\n'.join(train_ids) + '\n')
    (output_dir / 'val.txt').write_text('\n'.join(val_ids) + '\n')

    summary = dict(
        source=str(source),
        seed=seed,
        val_ratio=val_ratio,
        train_count=len(train_ids),
        val_count=len(val_ids),
        group_counts=group_counts,
    )
    (output_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path, help='Original RTM train.txt')
    parser.add_argument('output_dir', type=Path, help='Directory for split files')
    parser.add_argument('--val-ratio', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=20260903)
    return parser.parse_args()


def main():
    args = parse_args()
    summary = create_split(
        args.source, args.output_dir, val_ratio=args.val_ratio, seed=args.seed)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
