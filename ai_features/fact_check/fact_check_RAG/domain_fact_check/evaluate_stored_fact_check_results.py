from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

from src.xlsx_utils import load_xlsx_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate stored fact_check_results JSON from news_result.xlsx")
    parser.add_argument("--input", required=True, help="Path to xlsx file")
    parser.add_argument("--sheet", default="", help="Optional sheet name")
    parser.add_argument("--limit", type=int, default=0, help="Optional row limit")
    parser.add_argument("--output-json", default="", help="Optional aggregate output path")
    parser.add_argument("--output-jsonl", default="", help="Optional per-row output path")
    return parser.parse_args()


def load_rows(path: str, sheet_name: str = "", limit: int = 0) -> list[dict[str, Any]]:
    return load_xlsx_rows(path=path, sheet_name=sheet_name, limit=limit)


def safe_ratio(num: int | float, denom: int | float) -> float:
    return 0.0 if denom == 0 else float(num) / float(denom)


def parse_fact_check_results(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def get_external_issues(payload: dict[str, Any]) -> list[dict[str, Any]]:
    analysis = payload.get("analysis", {})
    issues = analysis.get("issues", [])
    return [issue for issue in issues if issue.get("check_type") == "external_fact"]


def get_internal_issues(payload: dict[str, Any]) -> list[dict[str, Any]]:
    analysis = payload.get("analysis", {})
    issues = analysis.get("issues", [])
    return [issue for issue in issues if issue.get("check_type") == "internal_quality"]


def flatten_evidence_documents(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return payload.get("evidence_documents", []) or []


def compute_proxy_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    total_rows = len(records)
    parsed_rows = [record for record in records if record["parsed_ok"]]
    truncated_rows = [record for record in records if record["raw_length"] >= 32767]

    domain_counter: Counter[str] = Counter()
    verdict_counter: Counter[str] = Counter()
    source_type_counter: Counter[str] = Counter()
    issue_label_counter: Counter[str] = Counter()

    claim_counts: list[int] = []
    verifiable_claim_counts: list[int] = []
    evidence_doc_counts: list[int] = []
    external_issue_counts: list[int] = []
    internal_issue_counts: list[int] = []

    articles_with_external = 0
    articles_with_evidence = 0
    articles_with_trusted_evidence = 0
    articles_with_supported = 0
    articles_with_contradicted = 0
    articles_with_unverified = 0
    articles_with_misleading = 0
    completeness_ok = 0

    total_evidence_docs = 0
    trusted_evidence_docs = 0

    per_article: list[dict[str, Any]] = []

    required_top_keys = {"domain", "analysis", "report_payload", "report_text"}

    for record in parsed_rows:
        payload = record["payload"]
        domain = payload.get("domain", "unknown")
        analysis = payload.get("analysis", {})
        summary = analysis.get("summary", {})
        claims = analysis.get("claims", [])
        evidence_docs = flatten_evidence_documents(payload)
        external_issues = get_external_issues(payload)
        internal_issues = get_internal_issues(payload)
        verdicts = [issue.get("verdict", "unknown") for issue in external_issues]

        domain_counter[domain] += 1
        for verdict in verdicts:
            verdict_counter[verdict] += 1
        for issue in internal_issues:
            issue_label_counter[issue.get("label", "unknown")] += 1

        claim_count = int(analysis.get("claim_count", len(claims) or 0))
        verifiable_claim_count = int(summary.get("verifiable_claims", 0))
        claim_counts.append(claim_count)
        verifiable_claim_counts.append(verifiable_claim_count)
        evidence_doc_counts.append(len(evidence_docs))
        external_issue_counts.append(len(external_issues))
        internal_issue_counts.append(len(internal_issues))

        if external_issues:
            articles_with_external += 1
        if evidence_docs:
            articles_with_evidence += 1
        if any(doc.get("source_type") in {"trusted_source", "official_site"} for doc in evidence_docs):
            articles_with_trusted_evidence += 1
        if "supported" in verdicts:
            articles_with_supported += 1
        if "contradicted" in verdicts:
            articles_with_contradicted += 1
        if "unverified" in verdicts:
            articles_with_unverified += 1
        if "misleading" in verdicts:
            articles_with_misleading += 1
        if required_top_keys.issubset(set(payload.keys())):
            completeness_ok += 1

        total_evidence_docs += len(evidence_docs)
        for doc in evidence_docs:
            source_type = doc.get("source_type", "unknown")
            source_type_counter[source_type] += 1
            if source_type in {"trusted_source", "official_site"}:
                trusted_evidence_docs += 1

        per_article.append(
            {
                "idx": record["idx"],
                "title": record["title"],
                "domain": domain,
                "claim_count": claim_count,
                "verifiable_claim_count": verifiable_claim_count,
                "evidence_doc_count": len(evidence_docs),
                "external_issue_count": len(external_issues),
                "internal_issue_count": len(internal_issues),
                "article_level_flags": {
                    "has_supported": "supported" in verdicts,
                    "has_contradicted": "contradicted" in verdicts,
                    "has_unverified": "unverified" in verdicts,
                    "has_misleading": "misleading" in verdicts,
                    "has_trusted_evidence": any(doc.get("source_type") in {"trusted_source", "official_site"} for doc in evidence_docs),
                },
            }
        )

    aggregate = {
        "total_rows": total_rows,
        "fact_check_present_rows": len([record for record in records if record["raw_value_present"]]),
        "fact_check_present_rate": round(safe_ratio(len([record for record in records if record["raw_value_present"]]), total_rows), 4),
        "likely_truncated_rows": len(truncated_rows),
        "likely_truncated_rate": round(safe_ratio(len(truncated_rows), total_rows), 4),
        "parse_success_rows": len(parsed_rows),
        "parse_success_rate": round(safe_ratio(len(parsed_rows), total_rows), 4),
        "json_completeness_rate": round(safe_ratio(completeness_ok, len(parsed_rows)), 4) if parsed_rows else 0.0,
        "domain_distribution": dict(domain_counter),
        "external_verdict_distribution": dict(verdict_counter),
        "evidence_source_type_distribution": dict(source_type_counter),
        "internal_issue_distribution": dict(issue_label_counter),
        "avg_claim_count": round(mean(claim_counts), 4) if claim_counts else 0.0,
        "avg_verifiable_claim_count": round(mean(verifiable_claim_counts), 4) if verifiable_claim_counts else 0.0,
        "avg_verifiable_claim_ratio": round(mean([safe_ratio(v, c) for v, c in zip(verifiable_claim_counts, claim_counts)]), 4) if claim_counts else 0.0,
        "avg_evidence_doc_count": round(mean(evidence_doc_counts), 4) if evidence_doc_counts else 0.0,
        "avg_external_issue_count": round(mean(external_issue_counts), 4) if external_issue_counts else 0.0,
        "avg_internal_issue_count": round(mean(internal_issue_counts), 4) if internal_issue_counts else 0.0,
        "evidence_coverage_rate": round(safe_ratio(articles_with_evidence, len(parsed_rows)), 4) if parsed_rows else 0.0,
        "trusted_evidence_article_rate": round(safe_ratio(articles_with_trusted_evidence, len(parsed_rows)), 4) if parsed_rows else 0.0,
        "trusted_evidence_doc_rate": round(safe_ratio(trusted_evidence_docs, total_evidence_docs), 4) if total_evidence_docs else 0.0,
        "external_verdict_article_rates": {
            "supported_article_rate": round(safe_ratio(articles_with_supported, len(parsed_rows)), 4) if parsed_rows else 0.0,
            "contradicted_article_rate": round(safe_ratio(articles_with_contradicted, len(parsed_rows)), 4) if parsed_rows else 0.0,
            "unverified_article_rate": round(safe_ratio(articles_with_unverified, len(parsed_rows)), 4) if parsed_rows else 0.0,
            "misleading_article_rate": round(safe_ratio(articles_with_misleading, len(parsed_rows)), 4) if parsed_rows else 0.0,
        },
        "notes": [
            "These are proxy metrics from stored fact_check_results JSON.",
            "True accuracy/F1 still requires human-labeled gold verdicts.",
            "High trusted_evidence rates and high parse/completeness rates support implementation reliability, not truth accuracy.",
        ],
    }

    return {
        "aggregate": aggregate,
        "per_article": per_article,
    }


def main() -> None:
    args = parse_args()
    rows = load_rows(args.input, sheet_name=args.sheet, limit=args.limit)
    records: list[dict[str, Any]] = []
    for row in rows:
        raw_value = row.get("fact_check_results")
        parsed = parse_fact_check_results(raw_value)
        records.append(
            {
                "idx": row.get("idx"),
                "title": row.get("title"),
                "raw_value_present": raw_value is not None and str(raw_value).strip() != "",
                "raw_length": len(str(raw_value)) if raw_value is not None else 0,
                "parsed_ok": parsed is not None,
                "payload": parsed,
            }
        )

    result = compute_proxy_metrics(records)
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.output_jsonl:
        with Path(args.output_jsonl).open("w", encoding="utf-8") as handle:
            for item in result["per_article"]:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(json.dumps(result["aggregate"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
