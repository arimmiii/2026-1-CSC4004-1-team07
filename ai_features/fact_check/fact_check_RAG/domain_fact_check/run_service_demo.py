from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if parent.name == "domain_fact_check")
sys.path.insert(0, str(PROJECT_ROOT))

from src.search_adapter import InMemorySearchAdapter
from src.service import run_fact_check_service
from src.utils import load_json, load_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run domain fact-check service with a local mock search adapter")
    parser.add_argument("--title", required=True)
    parser.add_argument("--body", default="")
    parser.add_argument("--body-file", default="")
    parser.add_argument("--category", default=None)
    parser.add_argument("--search-json", required=True, help="JSON object mapping query -> list[raw result]")
    parser.add_argument("--pretty", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    body = load_text(args.body_file) if args.body_file else args.body
    results_by_query = load_json(args.search_json)
    if not isinstance(results_by_query, dict):
        raise ValueError("search-json must be a JSON object mapping query strings to result lists.")

    payload = run_fact_check_service(
        title=args.title,
        body=body,
        category=args.category,
        search_adapter=InMemorySearchAdapter(results_by_query=results_by_query),
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2 if args.pretty else None))


if __name__ == "__main__":
    main()
