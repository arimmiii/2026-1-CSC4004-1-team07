from __future__ import annotations

import argparse

import joblib


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict clickbait label with trained linear SVM")
    parser.add_argument("--model", default="models/linear_svm_clickbait_concat.joblib", help="Path to model")
    parser.add_argument("--title", required=True, help="Title input")
    parser.add_argument("--body", default="", help="Body/content input")
    parser.add_argument(
        "--sep-token",
        default=" [SEP] ",
        help="Separator used to join title and body",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = joblib.load(args.model)
    parts = [args.title.strip(), args.body.strip()]
    input_text = args.sep_token.join([p for p in parts if p])

    pred = model.predict([input_text])[0]
    print(pred)


if __name__ == "__main__":
    main()
