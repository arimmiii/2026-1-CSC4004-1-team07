from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if parent.name == "domain_fact_check")
sys.path.insert(0, str(PROJECT_ROOT))

from src import EvidenceDocument, analyze_article
from src.models import SUPPORTED_DOMAINS
from src.utils import load_json, load_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze news article claims for domain-scoped fact checks")
    parser.add_argument("--title", required=True, help="Article title")
    parser.add_argument("--body", default="", help="Article body text")
    parser.add_argument("--body-file", default="", help="Path to body text file")
    parser.add_argument(
        "--domain",
        choices=SUPPORTED_DOMAINS,
        default=None,
        help="Force one of the supported domains",
    )
    parser.add_argument(
        "--evidence-json",
        default="",
        help="Optional JSON path with evidence docs: [{title, text, url, source_type, domain, published_at}]",
    )
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output")
    return parser.parse_args()


def load_body(args: argparse.Namespace) -> str:
    if args.body_file:
        return load_text(args.body_file)
    return args.body


def load_evidence(path: str) -> list[EvidenceDocument]:
    if not path:
        return []
    payload = load_json(path)
    if not isinstance(payload, list):
        raise ValueError("Evidence JSON must be a list of objects.")
    return [EvidenceDocument(**item) for item in payload]


def main() -> None:
    args = parse_args()
    body = load_body(args)
    evidence_docs = load_evidence(args.evidence_json)
    analysis = analyze_article(
        title=args.title,
        body=body,
        domain=args.domain,
        evidence_docs=evidence_docs,
    )
    if args.pretty:
        print(json.dumps(analysis.to_dict(), ensure_ascii=False, indent=2))
        return
    print(json.dumps(analysis.to_dict(), ensure_ascii=False))


if __name__ == "__main__":
    main()
