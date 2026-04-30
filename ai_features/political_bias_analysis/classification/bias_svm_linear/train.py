from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib

PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if parent.name == "political_bias_analysis")
sys.path.insert(0, str(PROJECT_ROOT))

from src.common import print_classification_metrics, load_training_data
from classification.bias_svm_linear.src.model import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TF-IDF + Linear SVM political bias classifier")
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--valid-data", required=True)
    parser.add_argument("--test-data", required=True)
    parser.add_argument("--text-col", default="text")
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--model-out", default="models/bias_svm_linear.joblib")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    x_train, y_train = load_training_data(args.train_data, args.text_col, args.label_col)
    x_valid, y_valid = load_training_data(args.valid_data, args.text_col, args.label_col)
    x_test, y_test = load_training_data(args.test_data, args.text_col, args.label_col)

    print(f"Train samples: {len(x_train):,}")
    print(f"Valid samples: {len(x_valid):,}")
    print(f"Test samples: {len(x_test):,}")

    model = build_model(random_state=args.random_state, c=args.c)
    model.fit(x_train, y_train)

    valid_pred = model.predict(x_valid)
    print_classification_metrics("Validation", y_valid, valid_pred)

    test_pred = model.predict(x_test)
    print_classification_metrics("Test", y_test, test_pred)

    model_out = Path(args.model_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_out)
    print(f"\nSaved model -> {model_out}")


if __name__ == "__main__":
    main()
