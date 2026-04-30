from __future__ import annotations

import argparse

import joblib


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict clickbait label with TF-IDF + Logistic Regression")
    parser.add_argument("--model", default="models/logreg_tfidf.joblib")
    parser.add_argument("--title", required=True)
    parser.add_argument("--body", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = joblib.load(args.model)
    parts = [args.title.strip(), args.body.strip()]
    text = " [SEP] ".join([p for p in parts if p])

    pred = model.predict([text])[0]
    print(pred)


if __name__ == "__main__":
    main()
