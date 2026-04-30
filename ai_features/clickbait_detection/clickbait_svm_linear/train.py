from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path

import joblib
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split

from src.data_utils import load_training_data
from src.model import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train linear SVM clickbait classifier")
    parser.add_argument("--data", required=True, help="Path to train CSV")
    parser.add_argument("--valid-data", default=None, help="Optional validation CSV path")
    parser.add_argument("--test-data", default=None, help="Optional test CSV path")
    parser.add_argument("--title-col", default="title", help="Title column name")
    parser.add_argument("--body-col", default="body", help="Body/content column name")
    parser.add_argument("--label-col", default="label", help="Label column name")
    parser.add_argument(
        "--valid-size",
        type=float,
        default=0.2,
        help="Used only when --valid-data is not provided",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--model-out", default="models/linear_svm_clickbait.joblib")
    return parser.parse_args()


def evaluate_split(name: str, y_true, y_pred) -> None:
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    weighted_f1 = f1_score(y_true, y_pred, average="weighted")

    print(f"\n[{name}] Macro F1: {macro_f1:.4f}")
    print(f"[{name}] Weighted F1: {weighted_f1:.4f}")
    print(f"[{name}] Classification Report:")
    print(classification_report(y_true, y_pred, digits=4))


def log(message: str) -> None:
    print(message, flush=True)


def main() -> None:
    args = parse_args()

    log("[1/6] Loading training data...")
    x_train, y_train = load_training_data(
        args.data,
        title_col=args.title_col,
        body_col=args.body_col,
        label_col=args.label_col,
    )

    if args.valid_data:
        log("[2/6] Loading validation data...")
        x_valid, y_valid = load_training_data(
            args.valid_data,
            title_col=args.title_col,
            body_col=args.body_col,
            label_col=args.label_col,
        )
    else:
        log("[2/6] Splitting train/valid from training data...")
        x_train, x_valid, y_train, y_valid = train_test_split(
            x_train,
            y_train,
            test_size=args.valid_size,
            random_state=args.random_state,
            stratify=y_train,
        )

    print(f"Train samples: {len(x_train):,}")
    print(f"Valid samples: {len(x_valid):,}")

    log("[3/6] Building model...")
    model = build_model(random_state=args.random_state)
    log("[4/6] Fitting model...")
    fit_start = time.time()
    model.fit(x_train, y_train)
    fit_seconds = time.time() - fit_start
    log(f"[4/6] Training done. fit_time={fit_seconds:.1f}s")

    del x_train, y_train
    gc.collect()

    log("[5/6] Evaluating validation split...")
    valid_pred = model.predict(x_valid)
    evaluate_split("Validation", y_valid, valid_pred)
    del x_valid, y_valid, valid_pred
    gc.collect()

    if args.test_data:
        log("[6/6] Loading test data...")
        x_test, y_test = load_training_data(
            args.test_data,
            title_col=args.title_col,
            body_col=args.body_col,
            label_col=args.label_col,
        )
        print(f"Test samples: {len(x_test):,}")
        log("[6/6] Evaluating test split...")
        test_pred = model.predict(x_test)
        evaluate_split("Test", y_test, test_pred)
        del x_test, y_test, test_pred
        gc.collect()

    model_out = Path(args.model_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_out)
    print(f"\nSaved model -> {model_out}")


if __name__ == "__main__":
    main()
