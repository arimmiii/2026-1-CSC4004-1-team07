from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict clickbait label with fine-tuned transformer")
    parser.add_argument("--model-name", "--model-dir", dest="model_name", default="models/klue_roberta_clickbait_title_body", help="Local model folder or Hugging Face repo id")
    parser.add_argument("--title", required=True)
    parser.add_argument("--body", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    parts = [args.title.strip(), args.body.strip()]
    text = " [SEP] ".join([p for p in parts if p])

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name)
    model.eval()

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        logits = model(**inputs).logits

    pred = int(torch.argmax(logits, dim=-1).item())
    print(pred)


if __name__ == "__main__":
    main()
