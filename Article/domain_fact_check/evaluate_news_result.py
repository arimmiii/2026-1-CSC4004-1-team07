from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

from src.explanation import build_report_payload
from src.pipeline import analyze_article
from src.xlsx_utils import load_xlsx_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch-evaluate news_result.xlsx with domain_fact_check")
    parser.add_argument("--input", required=True, help="Path to xlsx file")
    parser.add_argument("--sheet", default="", help="Optional sheet name")
    parser.add_argument("--limit", type=int, default=0, help="Optional row limit for quick tests")
    parser.add_argument("--output-json", default="", help="Optional output JSON path")
    parser.add_argument("--output-jsonl", default="", help="Optional per-article JSONL path")
    parser.add_argument(
        "--category-map-json",
        default="",
        help="Optional mapping JSON path, e.g. {\"IT/과학\": \"science\"}",
    )
    return parser.parse_args()


def load_rows(path: str, sheet_name: str = "", limit: int = 0) -> list[dict[str, Any]]:
    return load_xlsx_rows(path=path, sheet_name=sheet_name, limit=limit)


def load_category_map(path: str) -> dict[str, str]:
    if not path:
        return {}
    return json.loads(Path(path).read_text(encoding="utf-8"))


def normalize_category(raw_category: Any, category_map: dict[str, str]) -> str | None:
    if raw_category is None:
        return None
    category = str(raw_category).strip()
    if not category:
        return None
    return category_map.get(category, category_map.get(category.lower(), None))


def article_level_verdict(summary: dict[str, Any]) -> str:
    verdicts = summary.get("external_verdicts", {})
    if verdicts.get("contradicted", 0):
        return "contradicted"
    if verdicts.get("misleading", 0):
        return "misleading"
    if verdicts.get("supported", 0):
        return "supported"
    if verdicts.get("unverified", 0):
        return "unverified"
    return "no_external_verdict"


def safe_ratio(num: int, denom: int) -> float:
    return 0.0 if denom == 0 else num / denom


def maybe_compute_accuracy(records: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = [record for record in records if record.get("gold_verdict")]
    if not eligible:
        return {
            "available": False,
            "reason": "gold_verdict column not found or empty; true accuracy/F1 cannot be computed without labels.",
        }

    correct = sum(1 for record in eligible if record["gold_verdict"] == record["predicted_verdict"])
    labels = sorted(set(record["gold_verdict"] for record in eligible) | set(record["predicted_verdict"] for record in eligible))
    confusion: dict[str, dict[str, int]] = {gold: {pred: 0 for pred in labels} for gold in labels}
    for record in eligible:
        confusion[record["gold_verdict"]][record["predicted_verdict"]] += 1

    return {
        "available": True,
        "labeled_count": len(eligible),
        "accuracy": round(safe_ratio(correct, len(eligible)), 4),
        "confusion_matrix": confusion,
    }


def main() -> None:
    args = parse_args()
    rows = load_rows(args.input, sheet_name=args.sheet, limit=args.limit)
    category_map = load_category_map(args.category_map_json)

    per_article: list[dict[str, Any]] = []
    claim_counts: list[int] = []
    verifiable_counts: list[int] = []
    internal_issue_counts: list[int] = []
    domain_counter: Counter[str] = Counter()
    verdict_counter: Counter[str] = Counter()
    internal_flag_counter: Counter[str] = Counter()

    for row in rows:
        title = str(row.get("title") or "").strip()
        body = str(row.get("content") or "").strip()
        if not title and not body:
            continue

        forced_domain = normalize_category(row.get("category"), category_map)
        analysis = analyze_article(title=title, body=body, domain=forced_domain)
        payload = build_report_payload(analysis)
        summary = analysis.summary
        predicted_verdict = article_level_verdict(summary)

        claim_counts.append(analysis.claim_count)
        verifiable_counts.append(int(summary.get("verifiable_claims", 0)))
        internal_issue_count = len([issue for issue in analysis.issues if issue.check_type == "internal_quality"])
        internal_issue_counts.append(internal_issue_count)
        domain_counter[analysis.domain] += 1
        verdict_counter[predicted_verdict] += 1
        for label, count in summary.get("internal_quality_flags", {}).items():
            internal_flag_counter[label] += count

        record = {
            "idx": row.get("idx"),
            "title": title,
            "raw_category": row.get("category"),
            "mapped_domain": forced_domain,
            "predicted_domain": analysis.domain,
            "claim_count": analysis.claim_count,
            "verifiable_claims": summary.get("verifiable_claims", 0),
            "predicted_verdict": predicted_verdict,
            "internal_issue_count": internal_issue_count,
            "report_payload": payload,
            "gold_verdict": row.get("gold_verdict"),
        }
        per_article.append(record)

    aggregate = {
        "article_count": len(per_article),
        "domain_distribution": dict(domain_counter),
        "article_level_verdict_distribution": dict(verdict_counter),
        "avg_claim_count": round(mean(claim_counts), 4) if claim_counts else 0.0,
        "avg_verifiable_claim_count": round(mean(verifiable_counts), 4) if verifiable_counts else 0.0,
        "avg_internal_issue_count": round(mean(internal_issue_counts), 4) if internal_issue_counts else 0.0,
        "internal_flag_distribution": dict(internal_flag_counter),
        "accuracy_metrics": maybe_compute_accuracy(per_article),
    }

    result = {
        "aggregate": aggregate,
        "per_article": per_article,
    }

    if args.output_json:
        Path(args.output_json).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.output_jsonl:
        with Path(args.output_jsonl).open("w", encoding="utf-8") as handle:
            for item in per_article:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(json.dumps(aggregate, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
