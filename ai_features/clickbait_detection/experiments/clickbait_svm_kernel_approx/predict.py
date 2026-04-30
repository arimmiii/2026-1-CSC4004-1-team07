from __future__ import annotations

import argparse

import joblib


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict clickbait label with kernel-approx SVM")
    parser.add_argument("--model", default="models/kernel_approx_svm.joblib")
    parser.add_argument("--text", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = joblib.load(args.model)
    pred = model.predict([args.text])[0]
    print(pred)


if __name__ == "__main__":
    main()
