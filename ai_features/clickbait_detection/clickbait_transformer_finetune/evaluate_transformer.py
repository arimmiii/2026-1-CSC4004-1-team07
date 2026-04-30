from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned transformer on a labeled CSV file")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--text-col", default="text")
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    df = pd.read_csv(args.data_path, usecols=lambda col: col in {"text", args.text_col, args.label_col})
    if args.text_col not in df.columns:
        raise ValueError(f"text column not found: {args.text_col}")
    if args.label_col not in df.columns:
        raise ValueError(f"label column not found: {args.label_col}")

    texts = df[args.text_col].fillna("").astype(str).tolist()
    labels = df[args.label_col].astype(int).to_numpy()

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    def collate_fn(batch_texts: list[str]) -> dict[str, torch.Tensor]:
        return tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt",
        )

    dataloader = DataLoader(texts, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    preds: list[int] = []
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits
            preds.extend(torch.argmax(logits, dim=-1).cpu().tolist())

    preds_array = np.asarray(preds)
    metrics = {
        "accuracy": accuracy_score(labels, preds_array),
        "macro_f1": f1_score(labels, preds_array, average="macro"),
        "weighted_f1": f1_score(labels, preds_array, average="weighted"),
    }

    print(f"Samples: {len(labels):,}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print(f"Weighted F1: {metrics['weighted_f1']:.4f}")


if __name__ == "__main__":
    main()
