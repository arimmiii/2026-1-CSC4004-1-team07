from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib

PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if parent.name == "political_bias_analysis")
sys.path.insert(0, str(PROJECT_ROOT))

from src.common import compose_input_text, score_to_bucket


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict political bias score with TF-IDF + Linear SVR")
    parser.add_argument("--model", default="models/bias_svr_linear.joblib")
    parser.add_argument("--text", default="")
    parser.add_argument("--title", default="")
    parser.add_argument("--content", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = joblib.load(args.model)
    text = compose_input_text(title=args.title, content=args.content, text=args.text)
    score = float(model.predict([text])[0])
    bucket, label_name = score_to_bucket(score)
    print(f"score={score:.4f}")
    print(f"class={bucket}")
    print(f"label={label_name}")


if __name__ == "__main__":
    main()
