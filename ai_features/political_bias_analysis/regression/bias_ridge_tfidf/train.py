from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib

PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if parent.name == "political_bias_analysis")
sys.path.insert(0, str(PROJECT_ROOT))

from src.common import print_regression_metrics, load_training_data
from regression.bias_ridge_tfidf.src.model import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TF-IDF + Ridge political bias regressor")
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--valid-data", required=True)
    parser.add_argument("--test-data", required=True)
    parser.add_argument("--text-col", default="text")
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--model-out", default="models/bias_ridge_tfidf.joblib")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    x_train, y_train = load_training_data(args.train_data, args.text_col, args.label_col)
    x_valid, y_valid = load_training_data(args.valid_data, args.text_col, args.label_col)
    x_test, y_test = load_training_data(args.test_data, args.text_col, args.label_col)

    print(f"Train samples: {len(x_train):,}")
    print(f"Valid samples: {len(x_valid):,}")
    print(f"Test samples: {len(x_test):,}")

    model = build_model(alpha=args.alpha)
    model.fit(x_train, y_train.astype(float))

    valid_pred = model.predict(x_valid)
    print_regression_metrics("Validation", y_valid, valid_pred)

    test_pred = model.predict(x_test)
    print_regression_metrics("Test", y_test, test_pred)

    model_out = Path(args.model_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_out)
    print(f"\nSaved model -> {model_out}")


if __name__ == "__main__":
    main()
