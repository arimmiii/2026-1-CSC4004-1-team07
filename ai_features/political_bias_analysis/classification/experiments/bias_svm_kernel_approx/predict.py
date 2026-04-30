from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib

PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if parent.name == "political_bias_analysis")
sys.path.insert(0, str(PROJECT_ROOT))

from src.common import compose_input_text, label_to_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict political bias class with kernel-approx SVM")
    parser.add_argument("--model", default="models/bias_svm_kernel_approx.joblib")
    parser.add_argument("--text", default="")
    parser.add_argument("--title", default="")
    parser.add_argument("--content", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = joblib.load(args.model)
    text = compose_input_text(title=args.title, content=args.content, text=args.text)
    pred = int(model.predict([text])[0])
    print(f"class={pred}")
    print(f"label={label_to_name(pred)}")


if __name__ == "__main__":
    main()
