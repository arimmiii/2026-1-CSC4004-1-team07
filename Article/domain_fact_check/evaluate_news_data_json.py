from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from evaluate_stored_fact_check_results import compute_proxy_metrics, parse_fact_check_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate news_data.json with stored fact_check_results")
    parser.add_argument("--input", required=True, help="Path to news_data.json")
    parser.add_argument("--limit", type=int, default=0, help="Optional item limit")
    parser.add_argument("--start-idx", type=int, default=1, help="Only include items whose idx is >= this value")
    parser.add_argument("--output-json", default="", help="Optional aggregate output path")
    parser.add_argument("--output-jsonl", default="", help="Optional per-item output path")
    parser.add_argument(
        "--allowed-categories",
        nargs="*",
        default=["사회", "경제", "IT/과학", "생활/문화"],
        help="Only include these raw category values. Default matches the 4 supported service categories.",
    )
    return parser.parse_args()


def load_items(path: str, limit: int = 0) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Input JSON must be a list of article objects.")
    if limit:
        return payload[:limit]
    return payload


def main() -> None:
    args = parse_args()
    items = load_items(args.input, limit=args.limit)
    allowed_categories = set(args.allowed_categories or [])
    records: list[dict[str, Any]] = []
    for item in items:
        item_idx = item.get("idx")
        try:
            numeric_idx = int(item_idx)
        except Exception:
            numeric_idx = 0
        if numeric_idx < args.start_idx:
            continue
        raw_category = str(item.get("category") or "").strip()
        if allowed_categories and raw_category not in allowed_categories:
            continue
        raw_value = item.get("fact_check_results")
        parsed = raw_value if isinstance(raw_value, dict) else parse_fact_check_results(raw_value)
        records.append(
            {
                "idx": item_idx,
                "title": item.get("title"),
                "raw_category": raw_category,
                "raw_value_present": raw_value is not None and str(raw_value).strip() != "",
                "raw_length": len(str(raw_value)) if raw_value is not None else 0,
                "parsed_ok": parsed is not None,
                "payload": parsed,
            }
        )

    result = compute_proxy_metrics(records)
    result["aggregate"]["allowed_categories"] = sorted(allowed_categories)
    result["aggregate"]["start_idx"] = args.start_idx
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.output_jsonl:
        with Path(args.output_jsonl).open("w", encoding="utf-8") as handle:
            for item in result["per_article"]:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(json.dumps(result["aggregate"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
