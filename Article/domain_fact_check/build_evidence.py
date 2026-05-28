from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if parent.name == "domain_fact_check")
sys.path.insert(0, str(PROJECT_ROOT))

from src.evidence_builder import build_evidence_documents, generate_search_queries
from src.pipeline import extract_claims, infer_domain
from src.utils import load_json, load_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize search results into evidence documents")
    parser.add_argument("--title", required=True)
    parser.add_argument("--body", default="")
    parser.add_argument("--body-file", default="")
    parser.add_argument("--domain", default=None)
    parser.add_argument("--search-json", required=True, help="Raw search result JSON list path")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", default="", help="Optional output evidence JSON path")
    parser.add_argument("--pretty", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    body = load_text(args.body_file) if args.body_file else args.body
    domain = infer_domain(title=args.title, body=body, preferred_domain=args.domain)
    claims = extract_claims(title=args.title, body=body, domain=domain)
    raw_results = load_json(args.search_json)
    if not isinstance(raw_results, list):
        raise ValueError("Search JSON must be a list of objects.")

    evidence_docs = build_evidence_documents(raw_results=raw_results, domain=domain, claims=claims, top_k=args.top_k)
    payload = {
        "domain": domain,
        "queries": generate_search_queries(title=args.title, claims=claims, domain=domain),
        "evidence": [doc.__dict__ for doc in evidence_docs],
    }

    serialized = json.dumps(payload, ensure_ascii=False, indent=2 if args.pretty else None)
    if args.output:
        Path(args.output).write_text(serialized, encoding="utf-8")
    print(serialized)


if __name__ == "__main__":
    main()
