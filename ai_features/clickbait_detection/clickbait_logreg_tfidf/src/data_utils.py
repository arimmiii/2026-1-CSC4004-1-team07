from __future__ import annotations

import pandas as pd


def _join_columns(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    parts = [df[col].fillna("").astype(str).str.strip() for col in cols if col in df.columns]
    if not parts:
        raise ValueError(f"none of columns exist: {cols}. available: {list(df.columns)}")
    return pd.Series(
        [
            " [SEP] ".join([part.iloc[i] for part in parts if part.iloc[i]])
            for i in range(len(df))
        ],
        index=df.index,
    )


def load_training_data(
    csv_path: str,
    title_col: str = "title",
    body_col: str = "body",
    label_col: str = "label",
    text_col: str = "text",
) -> tuple[pd.Series, pd.Series]:
    wanted_cols = {title_col, body_col, label_col, text_col}
    df = pd.read_csv(csv_path, usecols=lambda col: col in wanted_cols)

    if label_col not in df.columns:
        raise ValueError(f"label column '{label_col}' not found. available: {list(df.columns)}")

    if title_col in df.columns or body_col in df.columns:
        x = _join_columns(df, [title_col, body_col])
    elif text_col in df.columns:
        x = df[text_col].fillna("").astype(str)
    else:
        raise ValueError(
            f"none of text/title/body columns found. available: {list(df.columns)}"
        )

    y = df[label_col]
    return x, y
