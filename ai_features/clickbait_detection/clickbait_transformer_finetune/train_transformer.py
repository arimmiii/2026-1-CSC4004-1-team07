from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune Transformer for clickbait detection")
    parser.add_argument("--model-name", default="klue/roberta-base")
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--valid-data", required=True)
    parser.add_argument("--test-data", required=True)
    parser.add_argument("--title-col", default="title")
    parser.add_argument("--body-col", default="body")
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.0)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-valid-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--output-dir", default="outputs/klue_roberta_clickbait")
    parser.add_argument("--save-model-dir", default="models/klue_roberta_clickbait")
    return parser.parse_args()


def load_split(path: str, title_col: str, body_col: str, label_col: str) -> pd.DataFrame:
    wanted_cols = {title_col, body_col, label_col}
    df = pd.read_csv(path, usecols=lambda col: col in wanted_cols)
    if title_col not in df.columns:
        raise ValueError(f"title column not found: {title_col}")
    if body_col not in df.columns:
        raise ValueError(f"body column not found: {body_col}")
    if label_col not in df.columns:
        raise ValueError(f"label column not found: {label_col}")

    df = df[[title_col, body_col, label_col]].rename(
        columns={title_col: "title", body_col: "body", label_col: "label"}
    )
    df["title"] = df["title"].fillna("").astype(str)
    df["body"] = df["body"].fillna("").astype(str)
    return df


def maybe_limit_samples(df: pd.DataFrame, limit: int | None, seed: int) -> pd.DataFrame:
    if limit is None or limit >= len(df):
        return df
    return df.sample(n=limit, random_state=seed).reset_index(drop=True)


def to_dataset(df: pd.DataFrame) -> Dataset:
    return Dataset.from_pandas(df, preserve_index=False)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
        "weighted_f1": f1_score(labels, preds, average="weighted"),
    }


def main() -> None:
    args = parse_args()
    has_cuda = torch.cuda.is_available()
    print(f"CUDA available: {has_cuda}")
    if has_cuda:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        raise RuntimeError("GPU runtime is not enabled. Colab에서 런타임 유형을 GPU로 바꾸고 다시 실행하세요.")

    train_df = maybe_limit_samples(
        load_split(args.train_data, args.title_col, args.body_col, args.label_col),
        args.max_train_samples,
        args.seed,
    )
    valid_df = maybe_limit_samples(
        load_split(args.valid_data, args.title_col, args.body_col, args.label_col),
        args.max_valid_samples,
        args.seed,
    )
    test_df = maybe_limit_samples(
        load_split(args.test_data, args.title_col, args.body_col, args.label_col),
        args.max_test_samples,
        args.seed,
    )
    print(f"Train: {len(train_df):,} | Valid: {len(valid_df):,} | Test: {len(test_df):,}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

    train_ds = to_dataset(train_df)
    valid_ds = to_dataset(valid_df)
    test_ds = to_dataset(test_df)

    def tokenize_fn(batch):
        texts = []
        for title, body in zip(batch["title"], batch["body"]):
            parts = [str(title).strip(), str(body).strip()]
            texts.append(" [SEP] ".join([p for p in parts if p]))
        return tokenizer(texts, truncation=True, max_length=args.max_length)

    train_ds = train_ds.map(tokenize_fn, batched=True)
    valid_ds = valid_ds.map(tokenize_fn, batched=True)
    test_ds = test_ds.map(tokenize_fn, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    ta_sig = inspect.signature(TrainingArguments.__init__).parameters
    use_bf16 = bool(has_cuda and hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported())
    use_fp16 = bool(has_cuda and not use_bf16)
    print(f"Mixed precision: bf16={use_bf16}, fp16={use_fp16}")

    training_kwargs = {
        "output_dir": args.output_dir,
        "save_strategy": "epoch",
        "logging_strategy": "steps",
        "logging_steps": 50,
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "num_train_epochs": args.epochs,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "load_best_model_at_end": True,
        "metric_for_best_model": "macro_f1",
        "greater_is_better": True,
        "seed": args.seed,
        "report_to": "none",
        "dataloader_pin_memory": has_cuda,
        "bf16": use_bf16,
        "fp16": use_fp16,
        "disable_tqdm": False,
    }
    training_kwargs["eval_strategy" if "eval_strategy" in ta_sig else "evaluation_strategy"] = "epoch"
    training_args = TrainingArguments(**training_kwargs)

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_ds,
        "eval_dataset": valid_ds,
        "data_collator": collator,
        "compute_metrics": compute_metrics,
    }
    trainer_sig = inspect.signature(Trainer.__init__).parameters
    if "tokenizer" in trainer_sig:
        trainer_kwargs["tokenizer"] = tokenizer
    elif "processing_class" in trainer_sig:
        trainer_kwargs["processing_class"] = tokenizer

    trainer = Trainer(**trainer_kwargs)
    trainer.train()

    valid_metrics = trainer.evaluate(valid_ds, metric_key_prefix="valid")
    test_metrics = trainer.evaluate(test_ds, metric_key_prefix="test")
    print("Validation metrics:")
    print(valid_metrics)
    print("Test metrics:")
    print(test_metrics)

    save_dir = Path(args.save_model_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(save_dir))
    tokenizer.save_pretrained(str(save_dir))

    metrics_path = save_dir / "metrics.json"
    metrics_payload = {"valid": valid_metrics, "test": test_metrics}
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, ensure_ascii=False, indent=2)

    print(f"Saved model -> {save_dir}")
    print(f"Saved metrics -> {metrics_path}")


if __name__ == "__main__":
    main()
