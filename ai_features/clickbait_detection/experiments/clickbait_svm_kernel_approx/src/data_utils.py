from __future__ import annotations

import pandas as pd


def load_training_data(csv_path: str, text_col: str = "text", label_col: str = "label") -> tuple[pd.Series, pd.Series]:
    df = pd.read_csv(csv_path)

    if text_col not in df.columns:
        raise ValueError(f"text column '{text_col}' not found. available: {list(df.columns)}")
    if label_col not in df.columns:
        raise ValueError(f"label column '{label_col}' not found. available: {list(df.columns)}")

    x = df[text_col].fillna("").astype(str)
    y = df[label_col]
    return x, y
