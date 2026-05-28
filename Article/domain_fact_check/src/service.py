from __future__ import annotations

from dataclasses import asdict
import logging

from .evidence_builder import build_evidence_documents, generate_search_queries
from .explanation import build_report, build_report_payload
from .llm_adapter import LLMEnhancer
from .logging_utils import log_json_event, setup_factcheck_logger
from .models import EvidenceDocument
from .pipeline import analyze_article, extract_claims, infer_domain
from .search_adapter import SearchAdapter, SearchRequest


def run_fact_check_service(
    *,
    title: str,
    body: str,
    category: str | None,
    article_url: str = "",
    published_at: str = "",
    search_adapter: SearchAdapter,
    search_top_k: int = 5,
    llm_enhancer: LLMEnhancer | None = None,
    logger: logging.Logger | None = None,
    log_path: str | None = None,
) -> dict[str, object]:
    active_logger = logger or setup_factcheck_logger(log_path=log_path)
    log_json_event(active_logger, "factcheck_start", {"title": title, "category": category, "body_chars": len(body), "search_top_k": search_top_k, "published_at": published_at, "article_url": article_url})
    domain = infer_domain(title=title, body=body, preferred_domain=category)
    log_json_event(active_logger, "domain_inferred", {"domain": domain})
    if llm_enhancer is not None:
        if getattr(llm_enhancer, "logger", None) is None:
            try:
                llm_enhancer.logger = active_logger
            except Exception:
                pass
        claims = llm_enhancer.extract_claims(title=title, body=body, domain=domain, context_date=published_at)
    else:
        claims = extract_claims(title=title, body=body, domain=domain, context_date=published_at)
    log_json_event(active_logger, "claims_ready", {"claim_count": len(claims), "verifiable_count": sum(1 for c in claims if c.verifiable)})
    queries = generate_search_queries(title=title, claims=claims, domain=domain)
    log_json_event(active_logger, "queries_generated", {"query_count": len(queries), "queries": queries})

    raw_results: list[dict] = []
    for query in queries:
        request = SearchRequest(query=query, domain=domain, top_k=search_top_k)
        results = search_adapter.search(request)
        raw_results.extend(results)
        log_json_event(active_logger, "search_query_done", {"query": query, "result_count": len(results)})

    evidence_docs = build_evidence_documents(
        raw_results=raw_results,
        domain=domain,
        claims=claims,
        top_k=search_top_k,
        excluded_url=article_url,
        article_title=title,
    )
    log_json_event(active_logger, "evidence_ranked", {"raw_result_count": len(raw_results), "evidence_doc_count": len(evidence_docs), "top_titles": [doc.title for doc in evidence_docs[:5]]})

    analysis = analyze_article(
        title=title,
        body=body,
        domain=domain,
        evidence_docs=evidence_docs,
        context_date=published_at,
    )
    log_json_event(active_logger, "analysis_done", {"claim_count": analysis.claim_count, "issue_count": len(analysis.issues), "summary": analysis.summary})
    if llm_enhancer is not None:
        llm_internal_issues = llm_enhancer.review_internal_quality(claims=claims, body=body, domain=domain)
        analysis.issues.extend(llm_internal_issues)
        analysis.summary["internal_quality_flags"] = {
            issue.label: sum(1 for current in analysis.issues if current.check_type == "internal_quality" and current.label == issue.label)
            for issue in llm_internal_issues
        } | analysis.summary.get("internal_quality_flags", {})
        llm_report_payload = llm_enhancer.build_user_report(analysis)
        report_payload = build_report_payload(analysis) | {"llm_report": llm_report_payload}
        detailed_report = build_report(analysis)
        issue_explanations = llm_report_payload.get("issue_explanations") or []
        report_text = llm_report_payload["user_summary"]
        if issue_explanations:
            report_text = "\n".join([report_text, "", "세부 설명:"] + [f"- {item}" for item in issue_explanations])
        report_text = "\n\n".join([report_text, detailed_report])
        log_json_event(active_logger, "llm_report_attached", {"has_llm_report": True, "internal_issue_count": len(llm_internal_issues)})
    else:
        report_payload = build_report_payload(analysis)
        report_text = build_report(analysis)
    log_json_event(active_logger, "factcheck_done", {"domain": domain, "report_text_chars": len(report_text)})

    return {
        "domain": domain,
        "queries": queries,
        "raw_search_results": raw_results,
        "evidence_documents": [asdict(doc) for doc in evidence_docs],
        "analysis": analysis.to_dict(),
        "report_payload": report_payload,
        "report_text": report_text,
    }


def run_fact_check_with_evidence(
    *,
    title: str,
    body: str,
    category: str | None,
    published_at: str = "",
    evidence_documents: list[EvidenceDocument],
    llm_enhancer: LLMEnhancer | None = None,
    logger: logging.Logger | None = None,
    log_path: str | None = None,
) -> dict[str, object]:
    active_logger = logger or setup_factcheck_logger(log_path=log_path)
    log_json_event(active_logger, "factcheck_with_evidence_start", {"title": title, "category": category, "body_chars": len(body), "evidence_doc_count": len(evidence_documents), "published_at": published_at})
    domain = infer_domain(title=title, body=body, preferred_domain=category)
    log_json_event(active_logger, "domain_inferred", {"domain": domain})
    if llm_enhancer is not None:
        if getattr(llm_enhancer, "logger", None) is None:
            try:
                llm_enhancer.logger = active_logger
            except Exception:
                pass
        claims = llm_enhancer.extract_claims(title=title, body=body, domain=domain, context_date=published_at)
    else:
        claims = extract_claims(title=title, body=body, domain=domain, context_date=published_at)
    log_json_event(active_logger, "claims_ready", {"claim_count": len(claims), "verifiable_count": sum(1 for c in claims if c.verifiable)})
    analysis = analyze_article(
        title=title,
        body=body,
        domain=domain,
        evidence_docs=evidence_documents,
        context_date=published_at,
    )
    log_json_event(active_logger, "analysis_done", {"claim_count": analysis.claim_count, "issue_count": len(analysis.issues), "summary": analysis.summary})
    if llm_enhancer is not None:
        analysis.claims = claims
        analysis.claim_count = len(claims)
        analysis.issues.extend(llm_enhancer.review_internal_quality(claims=claims, body=body, domain=domain))
        llm_report_payload = llm_enhancer.build_user_report(analysis)
        report_payload = build_report_payload(analysis) | {"llm_report": llm_report_payload}
        detailed_report = build_report(analysis)
        issue_explanations = llm_report_payload.get("issue_explanations") or []
        report_text = llm_report_payload["user_summary"]
        if issue_explanations:
            report_text = "\n".join([report_text, "", "세부 설명:"] + [f"- {item}" for item in issue_explanations])
        report_text = "\n\n".join([report_text, detailed_report])
    else:
        report_payload = build_report_payload(analysis)
        report_text = build_report(analysis)
    log_json_event(active_logger, "factcheck_with_evidence_done", {"domain": domain, "report_text_chars": len(report_text)})
    return {
        "domain": domain,
        "evidence_documents": [asdict(doc) for doc in evidence_documents],
        "analysis": analysis.to_dict(),
        "report_payload": report_payload,
        "report_text": report_text,
    }
