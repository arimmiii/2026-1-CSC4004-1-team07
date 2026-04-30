from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score, mean_absolute_error, mean_squared_error

BIAS_LABEL_NAMES = {
    1: "liberal",
    2: "lean_liberal",
    3: "neutral",
    4: "lean_conservative",
    5: "conservative",
}


def compose_input_text(title: str = "", content: str = "", text: str = "") -> str:
    if text and text.strip():
        return text.strip()

    parts = [str(title or "").strip(), str(content or "").strip()]
    return " [SEP] ".join(part for part in parts if part)


def label_to_name(label: int) -> str:
    return BIAS_LABEL_NAMES.get(int(label), f"class_{label}")


def score_to_bucket(score: float) -> tuple[int, str]:
    clipped = min(5.0, max(1.0, float(score)))
    rounded = int(round(clipped))
    rounded = min(5, max(1, rounded))
    return rounded, label_to_name(rounded)


def load_split_frame(csv_path: str | Path) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def load_training_data(csv_path: str | Path, text_col: str = "text", label_col: str = "label") -> tuple[pd.Series, pd.Series]:
    df = load_split_frame(csv_path)

    if text_col not in df.columns:
        raise ValueError(f"text column '{text_col}' not found. available: {list(df.columns)}")
    if label_col not in df.columns:
        raise ValueError(f"label column '{label_col}' not found. available: {list(df.columns)}")

    x = df[text_col].fillna("").astype(str)
    y = df[label_col]
    return x, y


def print_classification_metrics(name: str, y_true, y_pred) -> tuple[float, float, float]:
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    weighted_f1 = f1_score(y_true, y_pred, average="weighted")
    acc = accuracy_score(y_true, y_pred)

    print(f"\n[{name}] Macro F1: {macro_f1:.4f}")
    print(f"[{name}] Weighted F1: {weighted_f1:.4f}")
    print(f"[{name}] Accuracy: {acc:.4f}")
    print(f"[{name}] Classification Report:")
    print(classification_report(y_true, y_pred, digits=4))

    return macro_f1, weighted_f1, acc


def print_regression_metrics(name: str, y_true, y_pred) -> tuple[float, float, float, float, float]:
    y_true_series = pd.Series(y_true).astype(float)
    y_pred_series = pd.Series(y_pred).astype(float)

    mae = mean_absolute_error(y_true_series, y_pred_series)
    rmse = math.sqrt(mean_squared_error(y_true_series, y_pred_series))
    pearson = y_true_series.corr(y_pred_series, method="pearson")
    spearman = y_true_series.corr(y_pred_series, method="spearman")

    rounded_pred = y_pred_series.clip(1.0, 5.0).round().astype(int)
    rounded_true = y_true_series.round().astype(int)
    macro_f1 = f1_score(rounded_true, rounded_pred, average="macro")
    acc = accuracy_score(rounded_true, rounded_pred)

    print(f"\n[{name}] MAE: {mae:.4f}")
    print(f"[{name}] RMSE: {rmse:.4f}")
    print(f"[{name}] Pearson: {pearson:.4f}")
    print(f"[{name}] Spearman: {spearman:.4f}")
    print(f"[{name}] Rounded Macro F1: {macro_f1:.4f}")
    print(f"[{name}] Rounded Accuracy: {acc:.4f}")
    print(f"[{name}] Prediction Range: {y_pred_series.min():.4f} ~ {y_pred_series.max():.4f}")

    return mae, rmse, pearson, spearman, macro_f1
