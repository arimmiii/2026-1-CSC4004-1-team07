from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate classification model with accuracy/macro-f1/top2-accuracy")
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--text-col", default="text")
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--label-offset", type=int, default=0, help="Add this offset to predicted labels/classes")
    return parser.parse_args()


def compute_top2_accuracy(model, x: pd.Series, y_true: np.ndarray, label_offset: int = 0) -> float | None:
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(x)
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(x)
    else:
        return None

    scores = np.asarray(scores)
    if scores.ndim == 1:
        return None

    # Convert 1D class labels to class index positions.
    classes = np.asarray(model.classes_, dtype=int) if hasattr(model, "classes_") else None
    top2_idx = np.argsort(scores, axis=1)[:, -2:]
    if classes is not None:
        top2_labels = classes[top2_idx] + label_offset
    else:
        top2_labels = top2_idx + label_offset

    hit = (top2_labels == y_true.reshape(-1, 1)).any(axis=1)
    return float(hit.mean())


def main() -> None:
    args = parse_args()
    model = joblib.load(args.model)
    df = pd.read_csv(args.data)

    x = df[args.text_col].fillna("").astype(str)
    y = df[args.label_col].astype(int).to_numpy()
    y_pred = model.predict(x).astype(int) + args.label_offset

    acc = accuracy_score(y, y_pred)
    macro_f1 = f1_score(y, y_pred, average="macro")
    top2 = compute_top2_accuracy(model, x, y, label_offset=args.label_offset)

    print(f"model={Path(args.model).as_posix()}")
    print(f"data={Path(args.data).as_posix()}")
    print(f"accuracy={acc:.4f}")
    print(f"macro_f1={macro_f1:.4f}")
    if top2 is None:
        print("top2_accuracy=NA")
    else:
        print(f"top2_accuracy={top2:.4f}")


if __name__ == "__main__":
    main()
