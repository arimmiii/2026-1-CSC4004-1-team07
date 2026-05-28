from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_ALLOWED_CATEGORIES = ["사회", "경제", "IT/과학", "생활/문화"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare manual review files from news_data.json, excluding null rows and splitting JSON vs natural-language fact_check_results."
    )
    parser.add_argument("--input", required=True, help="Path to news_data.json")
    parser.add_argument("--start-idx", type=int, default=31, help="Only include rows whose idx is >= this value")
    parser.add_argument(
        "--allowed-categories",
        nargs="*",
        default=DEFAULT_ALLOWED_CATEGORIES,
        help="Only include these raw category values",
    )
    parser.add_argument(
        "--output-prefix",
        default="domain_fact_check/examples/manual_review",
        help="Prefix path for output files",
    )
    return parser.parse_args()


def load_items(path: str) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Input JSON must be a list.")
    return payload


def classify_fact_check_value(value: Any) -> str:
    if value is None:
        return "null"
    text = str(value).strip()
    if not text:
        return "null"
    try:
        json.loads(text)
        return "json"
    except json.JSONDecodeError:
        return "text"


def main() -> None:
    args = parse_args()
    items = load_items(args.input)
    allowed_categories = set(args.allowed_categories)

    json_cases: list[dict[str, Any]] = []
    text_cases: list[dict[str, Any]] = []

    for item in items:
        try:
            idx = int(item.get("idx", 0))
        except Exception:
            idx = 0
        if idx < args.start_idx:
            continue

        raw_category = str(item.get("category") or "").strip()
        if raw_category not in allowed_categories:
            continue

        result_type = classify_fact_check_value(item.get("fact_check_results"))
        if result_type == "null":
            continue

        record = {
            "idx": idx,
            "title": item.get("title"),
            "category": raw_category,
            "link": item.get("link"),
            "content": item.get("content"),
            "fact_check_results": item.get("fact_check_results"),
        }
        if result_type == "json":
            json_cases.append(record)
        else:
            text_cases.append(record)

    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    json_path = output_prefix.with_name(output_prefix.name + "_json_cases.json")
    text_path = output_prefix.with_name(output_prefix.name + "_text_cases.json")
    jsonl_path = output_prefix.with_name(output_prefix.name + "_rerun_candidates.jsonl")

    json_path.write_text(json.dumps(json_cases, ensure_ascii=False, indent=2), encoding="utf-8")
    text_path.write_text(json.dumps(text_cases, ensure_ascii=False, indent=2), encoding="utf-8")

    with jsonl_path.open("w", encoding="utf-8") as handle:
        for record in json_cases + text_cases:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = {
        "start_idx": args.start_idx,
        "allowed_categories": sorted(allowed_categories),
        "json_case_count": len(json_cases),
        "text_case_count": len(text_cases),
        "json_cases_path": str(json_path),
        "text_cases_path": str(text_path),
        "rerun_candidates_jsonl_path": str(jsonl_path),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
