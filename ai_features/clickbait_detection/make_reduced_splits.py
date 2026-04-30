from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path('/mnt/c/Users/JaeHong/Desktop/sw_project/ai_features/clickbait_detection')
RAW_ROOT = ROOT / 'raw data'
OUTPUT_DIR = ROOT / 'data'
COPY_TARGETS = [
    ROOT / 'clickbait_svm_linear' / 'data',
    ROOT / 'clickbait_logreg_tfidf' / 'data',
    ROOT / 'clickbait_transformer_finetune' / 'data',
]

TARGET_SIZES = {
    'train': 200_000,
    'valid': 25_000,
    'test': 25_000,
}

SPLIT_SEED_OFFSETS = {
    'train': 11,
    'valid': 23,
    'test': 37,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Create reduced title/body stratified splits from raw JSON data')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--raw-root', default=str(RAW_ROOT))
    parser.add_argument('--output-dir', default=str(OUTPUT_DIR))
    return parser.parse_args()


def list_json_files(raw_root: Path) -> list[Path]:
    files = sorted(raw_root.rglob('*.json'))
    if not files:
        raise FileNotFoundError(f'No JSON files found under {raw_root}')
    return files


def read_label(raw_json_path: Path) -> str:
    with raw_json_path.open('r', encoding='utf-8') as f:
        item = json.load(f)
    labeled_info = item.get('labeledDataInfo', {})
    if 'clickbaitClass' not in labeled_info:
        raise ValueError(f'missing clickbaitClass in {raw_json_path}')
    return str(int(labeled_info['clickbaitClass']))


def extract_fields(raw_json_path: Path) -> tuple[str, str, str]:
    with raw_json_path.open('r', encoding='utf-8') as f:
        item = json.load(f)

    source_info = item.get('sourceDataInfo', {})
    labeled_info = item.get('labeledDataInfo', {})

    title = str(labeled_info.get('newTitle') or source_info.get('newsTitle') or '').strip()
    body = str(source_info.get('newsContent') or '').strip()
    label = str(int(labeled_info.get('clickbaitClass', 0)))
    return title, body, label


def target_label_counts(total_target: int, label_counts: Counter[str]) -> dict[str, int]:
    total = sum(label_counts.values())
    if total == 0:
        raise ValueError('empty source corpus')

    labels = sorted(label_counts.keys())
    quotas = {label: total_target * (label_counts[label] / total) for label in labels}
    counts = {label: int(math.floor(quotas[label])) for label in labels}
    remainder = total_target - sum(counts.values())
    if remainder > 0:
        frac_order = sorted(labels, key=lambda l: (quotas[l] - counts[l], label_counts[l]), reverse=True)
        for label in frac_order[:remainder]:
            counts[label] += 1
    return counts


def build_assignments(raw_root: Path, seed: int) -> dict[str, list[Path]]:
    rng = random.Random(seed)
    buckets: dict[str, list[Path]] = defaultdict(list)

    files = list_json_files(raw_root)
    for i, json_path in enumerate(files, start=1):
        label = read_label(json_path)
        buckets[label].append(json_path)
        if i % 50000 == 0:
            print(f'[scan] {i:,} files processed')

    for label, paths in buckets.items():
        rng.shuffle(paths)
        print(f'[scan] label={label} count={len(paths):,}')

    total_counts = Counter({label: len(paths) for label, paths in buckets.items()})
    split_targets = {
        split: target_label_counts(size, total_counts)
        for split, size in TARGET_SIZES.items()
    }
    print('[scan] split targets per label:', split_targets)

    assignments: dict[str, list[Path]] = {split: [] for split in TARGET_SIZES}
    offsets = {label: 0 for label in buckets}
    for split in ['train', 'valid', 'test']:
        for label in sorted(buckets.keys()):
            k = split_targets[split][label]
            start = offsets[label]
            end = start + k
            assignments[split].extend(buckets[label][start:end])
            offsets[label] = end

    for label, paths in buckets.items():
        used = offsets[label]
        if used != len(paths):
            raise RuntimeError(f'label {label} unused paths={len(paths)-used}')

    return assignments


def write_split(split_name: str, raw_paths: list[Path], raw_root: Path, out_path: Path, seed: int) -> None:
    rng = random.Random(seed)
    rows = []

    for i, json_path in enumerate(raw_paths, start=1):
        title, body, label = extract_fields(json_path)
        rows.append((title, body, label))
        if i % 25000 == 0:
            print(f'[{split_name}] {i:,}/{len(raw_paths):,} rows extracted')

    rng.shuffle(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['title', 'body', 'label'])
        writer.writerows(rows)

    counts = Counter(label for _, _, label in rows)
    print(f'[{split_name}] saved {len(rows):,} rows -> {out_path} | label_counts={dict(counts)}')


def copy_output_to_targets(output_dir: Path) -> None:
    for target in COPY_TARGETS:
        target.mkdir(parents=True, exist_ok=True)
        for name in ['train.csv', 'valid.csv', 'test.csv']:
            src = output_dir / name
            dst = target / name
            dst.write_bytes(src.read_bytes())
        print(f'copied to {target}')


def main() -> None:
    args = parse_args()
    raw_root = Path(args.raw_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    assignments = build_assignments(raw_root, args.seed)
    for split, raw_paths in assignments.items():
        write_split(
            split_name=split,
            raw_paths=raw_paths,
            raw_root=raw_root,
            out_path=output_dir / f'{split}.csv',
            seed=args.seed + SPLIT_SEED_OFFSETS[split],
        )

    copy_output_to_targets(output_dir)
    print('done')


if __name__ == '__main__':
    main()
