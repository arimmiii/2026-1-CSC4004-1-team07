from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from src import OpenAILLMEnhancer, TavilySearchAdapter, SerpApiSearchAdapter, run_fact_check_service


DEFAULT_ALLOWED_CATEGORIES = ["사회", "경제", "IT/과학", "생활/문화"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rerun domain fact check on news_data.json and compute verdict metrics")
    parser.add_argument("--input", required=True, help="Path to news_data.json")
    parser.add_argument("--start-idx", type=int, default=31)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--allowed-categories",
        nargs="*",
        default=DEFAULT_ALLOWED_CATEGORIES,
        help="Only include these raw category values",
    )
    parser.add_argument("--search-provider", choices=("tavily", "serpapi"), default="tavily")
    parser.add_argument("--search-top-k", type=int, default=5)
    parser.add_argument("--llm-model", default="gpt-5.4-mini")
    parser.add_argument("--output-json", default="", help="Optional full output path")
    parser.add_argument("--output-jsonl", default="", help="Optional per-article output path")
    parser.add_argument("--log-path", default="", help="Optional pipeline log file path")
    return parser.parse_args()


def load_items(path: str, limit: int = 0) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Input JSON must be a list.")
    if limit:
        return payload[:limit]
    return payload


def choose_search_adapter(provider: str):
    if provider == "serpapi":
        return SerpApiSearchAdapter()
    return TavilySearchAdapter()


def normalize_category(raw_category: str) -> str:
    mapping = {
        "사회": "society",
        "경제": "economy",
        "IT/과학": "science",
        "생활/문화": "lifestyle_culture",
    }
    return mapping.get(raw_category, raw_category)


def article_level_verdict(payload: dict[str, Any]) -> str:
    issues = payload.get("analysis", {}).get("issues", []) or []
    verdicts = [issue.get("verdict") for issue in issues if issue.get("check_type") == "external_fact"]
    if "contradicted" in verdicts:
        return "contradicted"
    if "misleading" in verdicts:
        return "misleading"
    if "supported" in verdicts:
        return "supported"
    if "unverified" in verdicts:
        return "unverified"
    return "no_external_verdict"


def main() -> None:
    args = parse_args()
    items = load_items(args.input, limit=args.limit)
    allowed = set(args.allowed_categories)
    search = choose_search_adapter(args.search_provider)
    llm = OpenAILLMEnhancer(model=args.llm_model)

    results: list[dict[str, Any]] = []
    verdict_counter: Counter[str] = Counter()
    domain_counter: Counter[str] = Counter()

    for item in items:
        try:
            idx = int(item.get("idx", 0))
        except Exception:
            idx = 0
        raw_category = str(item.get("category") or "").strip()
        if idx < args.start_idx:
            continue
        if raw_category not in allowed:
            continue
        if item.get("fact_check_results") is None:
            continue

        payload = run_fact_check_service(
            title=str(item.get("title") or ""),
            body=str(item.get("content") or ""),
            category=normalize_category(raw_category),
            article_url=str(item.get("link") or ""),
            search_adapter=search,
            search_top_k=args.search_top_k,
            llm_enhancer=llm,
            log_path=args.log_path or None,
        )
        verdict = article_level_verdict(payload)
        verdict_counter[verdict] += 1
        domain_counter[payload.get("domain", "unknown")] += 1
        results.append(
            {
                "idx": idx,
                "title": item.get("title"),
                "raw_category": raw_category,
                "rerun_result": payload,
                "article_level_verdict": verdict,
            }
        )

    aggregate = {
        "start_idx": args.start_idx,
        "allowed_categories": sorted(allowed),
        "rerun_count": len(results),
        "domain_distribution": dict(domain_counter),
        "article_level_verdict_distribution": dict(verdict_counter),
    }
    output = {"aggregate": aggregate, "results": results}

    if args.output_json:
        Path(args.output_json).write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.output_jsonl:
        with Path(args.output_jsonl).open("w", encoding="utf-8") as handle:
            for item in results:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(json.dumps(aggregate, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
