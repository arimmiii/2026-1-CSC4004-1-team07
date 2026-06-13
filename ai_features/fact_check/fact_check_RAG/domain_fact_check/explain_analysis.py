from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if parent.name == "domain_fact_check")
sys.path.insert(0, str(PROJECT_ROOT))

from src.explanation import build_report, build_report_payload
from src.models import AnalysisIssue, ArticleAnalysis, Claim
from src.utils import load_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build user-facing explanation from analysis JSON")
    parser.add_argument("--analysis-json", required=True)
    parser.add_argument("--format", choices=("text", "json"), default="text")
    return parser.parse_args()


def _load_analysis(path: str) -> ArticleAnalysis:
    payload = load_json(path)
    claims = [Claim(**claim) for claim in payload["claims"]]
    issues = [AnalysisIssue(**issue) for issue in payload["issues"]]
    return ArticleAnalysis(
        domain=payload["domain"],
        title=payload["title"],
        claim_count=payload["claim_count"],
        claims=claims,
        issues=issues,
        summary=payload["summary"],
    )


def main() -> None:
    args = parse_args()
    analysis = _load_analysis(args.analysis_json)
    if args.format == "json":
        print(json.dumps(build_report_payload(analysis), ensure_ascii=False, indent=2))
        return
    print(build_report(analysis))


if __name__ == "__main__":
    main()
