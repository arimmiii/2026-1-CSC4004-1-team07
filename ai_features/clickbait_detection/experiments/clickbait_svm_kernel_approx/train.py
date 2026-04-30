from __future__ import annotations

import argparse
from pathlib import Path

import joblib
from sklearn.metrics import accuracy_score, classification_report, f1_score

from src.data_utils import load_training_data
from src.model import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train kernel-approx SVM clickbait classifier")
    parser.add_argument("--train-data", required=True, help="Path to train CSV")
    parser.add_argument("--valid-data", required=True, help="Path to validation CSV")
    parser.add_argument("--test-data", required=True, help="Path to test CSV")
    parser.add_argument("--text-col", default="text")
    parser.add_argument("--label-col", default="label")

    parser.add_argument("--svd-components", type=int, default=256)
    parser.add_argument("--rbf-components", type=int, default=512)
    parser.add_argument("--gamma", type=float, default=0.7)
    parser.add_argument("--c", type=float, default=1.0)

    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--model-out", default="models/kernel_approx_svm.joblib")
    return parser.parse_args()


def evaluate_split(name: str, y_true, y_pred) -> tuple[float, float, float]:
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    weighted_f1 = f1_score(y_true, y_pred, average="weighted")
    acc = accuracy_score(y_true, y_pred)

    print(f"\n[{name}] Macro F1: {macro_f1:.4f}")
    print(f"[{name}] Weighted F1: {weighted_f1:.4f}")
    print(f"[{name}] Accuracy: {acc:.4f}")
    print(f"[{name}] Classification Report:")
    print(classification_report(y_true, y_pred, digits=4))

    return macro_f1, weighted_f1, acc


def main() -> None:
    args = parse_args()

    x_train, y_train = load_training_data(args.train_data, args.text_col, args.label_col)
    x_valid, y_valid = load_training_data(args.valid_data, args.text_col, args.label_col)
    x_test, y_test = load_training_data(args.test_data, args.text_col, args.label_col)

    print(f"Train samples: {len(x_train):,}")
    print(f"Valid samples: {len(x_valid):,}")
    print(f"Test samples: {len(x_test):,}")

    model = build_model(
        random_state=args.random_state,
        svd_components=args.svd_components,
        rbf_components=args.rbf_components,
        gamma=args.gamma,
        c=args.c,
    )
    model.fit(x_train, y_train)

    valid_pred = model.predict(x_valid)
    evaluate_split("Validation", y_valid, valid_pred)

    test_pred = model.predict(x_test)
    evaluate_split("Test", y_test, test_pred)

    model_out = Path(args.model_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_out)
    print(f"\nSaved model -> {model_out}")


if __name__ == "__main__":
    main()
