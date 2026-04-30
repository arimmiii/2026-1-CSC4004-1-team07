from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib

PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if parent.name == "political_bias_analysis")
sys.path.insert(0, str(PROJECT_ROOT))

from src.common import print_classification_metrics, load_training_data
from classification.experiments.bias_gbdt_tfidf.src.model import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TF-IDF/SVD + GBDT political bias classifier")
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--valid-data", required=True)
    parser.add_argument("--test-data", required=True)
    parser.add_argument("--text-col", default="text")
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--model-type", choices=["lightgbm", "xgboost"], default="lightgbm")
    parser.add_argument("--svd-components", type=int, default=220)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--num-leaves", type=int, default=63)
    parser.add_argument("--word-max-features", type=int, default=120000)
    parser.add_argument("--char-max-features", type=int, default=180000)
    parser.add_argument("--no-char-features", action="store_true")
    parser.add_argument("--n-jobs", type=int, default=2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--model-out", default="models/bias_gbdt_tfidf.joblib")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    x_train, y_train = load_training_data(args.train_data, args.text_col, args.label_col)
    x_valid, y_valid = load_training_data(args.valid_data, args.text_col, args.label_col)
    x_test, y_test = load_training_data(args.test_data, args.text_col, args.label_col)

    print(f"Train samples: {len(x_train):,}")
    print(f"Valid samples: {len(x_valid):,}")
    print(f"Test samples: {len(x_test):,}")

    model = build_model(
        model_type=args.model_type,
        random_state=args.random_state,
        svd_components=args.svd_components,
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        num_leaves=args.num_leaves,
        word_max_features=args.word_max_features,
        char_max_features=args.char_max_features,
        n_jobs=args.n_jobs,
        use_char_features=not args.no_char_features,
        num_class=int(y_train.astype(int).nunique()),
    )
    if args.model_type == "xgboost":
        y_train_fit = y_train.astype(int) - 1
    else:
        y_train_fit = y_train

    model.fit(x_train, y_train_fit)

    valid_pred = model.predict(x_valid)
    if args.model_type == "xgboost":
        valid_pred = valid_pred.astype(int) + 1
    print_classification_metrics("Validation", y_valid, valid_pred)

    test_pred = model.predict(x_test)
    if args.model_type == "xgboost":
        test_pred = test_pred.astype(int) + 1
    print_classification_metrics("Test", y_test, test_pred)

    model_out = Path(args.model_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_out)
    print(f"\nSaved model -> {model_out}")


if __name__ == "__main__":
    main()
