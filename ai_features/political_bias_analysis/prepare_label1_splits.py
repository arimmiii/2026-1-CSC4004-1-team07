from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.common import compose_input_text, label_to_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare train/valid/test CSV splits for political bias label1 experiments")
    parser.add_argument("--train-csv", default="data/complete_train.csv")
    parser.add_argument("--test-csv", default="data/complete_test.csv")
    parser.add_argument("--out-dir", default="data/splits_label1")
    parser.add_argument("--valid-size", type=float, default=0.1)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--text-source",
        choices=["title", "content", "title_plus_content"],
        default="title_plus_content",
    )
    parser.add_argument("--label-col", default="label1")
    parser.add_argument(
        "--label-mode",
        choices=["five_class", "three_class"],
        default="five_class",
        help="five_class: keep label1 as 1~5, three_class: map 1-2/3/4-5 to 1/2/3",
    )
    return parser.parse_args()


def build_text(row: pd.Series, text_source: str) -> str:
    title = str(row.get("title") or "").strip()
    content = str(row.get("content") or "").strip()

    if text_source == "title":
        return title
    if text_source == "content":
        return content
    if text_source == "title_plus_content":
        return compose_input_text(title=title, content=content)

    raise ValueError(f"unsupported text_source: {text_source}")


def map_label(raw_label: int, label_mode: str) -> int:
    if label_mode == "five_class":
        return int(raw_label)
    if label_mode == "three_class":
        if raw_label <= 2:
            return 1
        if raw_label == 3:
            return 2
        return 3
    raise ValueError(f"unsupported label_mode: {label_mode}")


def label_name_for_mode(label: int, label_mode: str) -> str:
    if label_mode == "five_class":
        return label_to_name(label)
    if label_mode == "three_class":
        mapping = {1: "liberal_block", 2: "neutral", 3: "conservative_block"}
        return mapping.get(int(label), f"class_{label}")
    return f"class_{label}"


def normalize_frame(df: pd.DataFrame, text_source: str, label_col: str, label_mode: str) -> pd.DataFrame:
    required_cols = {"title", "content", "date", "article_url", label_col}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")

    out = df.copy()
    out["text"] = out.apply(lambda row: build_text(row, text_source), axis=1)
    out["label"] = out[label_col].astype(int).map(lambda x: map_label(int(x), label_mode))
    out["label_name"] = out["label"].map(lambda x: label_name_for_mode(int(x), label_mode))
    keep_cols = [col for col in ["seq", "title", "content", "date", "article_url"] if col in out.columns]
    keep_cols += ["text", "label", "label_name"]
    out = out[keep_cols]
    out = out[out["text"].str.strip() != ""].reset_index(drop=True)
    return out


def print_split_stats(name: str, split_df: pd.DataFrame, label_mode: str) -> None:
    print(f"{name}: {len(split_df):,}")
    label_counts = split_df["label"].value_counts().sort_index()
    for label, count in label_counts.items():
        pct = count / len(split_df) * 100 if len(split_df) else 0.0
        print(f"  {label} ({label_name_for_mode(int(label), label_mode)}): {count:,} ({pct:.2f}%)")


def main() -> None:
    args = parse_args()

    train_df = pd.read_csv(args.train_csv)
    test_df = pd.read_csv(args.test_csv)

    train_df = normalize_frame(train_df, args.text_source, args.label_col, args.label_mode)
    test_df = normalize_frame(test_df, args.text_source, args.label_col, args.label_mode)

    train_split, valid_split = train_test_split(
        train_df,
        test_size=args.valid_size,
        random_state=args.random_state,
        stratify=train_df["label"],
    )

    train_split = train_split.reset_index(drop=True)
    valid_split = valid_split.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = out_dir / "train.csv"
    valid_path = out_dir / "valid.csv"
    test_path = out_dir / "test.csv"

    train_split.to_csv(train_path, index=False, encoding="utf-8-sig")
    valid_split.to_csv(valid_path, index=False, encoding="utf-8-sig")
    test_df.to_csv(test_path, index=False, encoding="utf-8-sig")

    print("Split statistics")
    print_split_stats("Train", train_split, args.label_mode)
    print_split_stats("Valid", valid_split, args.label_mode)
    print_split_stats("Test", test_df, args.label_mode)

    print("\nSaved files")
    print(f"- {train_path}")
    print(f"- {valid_path}")
    print(f"- {test_path}")


if __name__ == "__main__":
    main()
