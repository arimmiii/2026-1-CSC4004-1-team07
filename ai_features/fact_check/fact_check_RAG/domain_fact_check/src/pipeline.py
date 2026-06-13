from __future__ import annotations

from collections import Counter

from .external_verifier import verify_claims_against_evidence
from .internal_quality import review_internal_quality
from .models import ArticleAnalysis, Claim, EvidenceDocument
from .taxonomy import CLAIM_TYPE_KEYWORDS, DOMAIN_RULES
from .utils import find_dates, find_entities, find_numbers, split_sentences


def infer_domain(title: str, body: str, preferred_domain: str | None = None) -> str:
    if preferred_domain in DOMAIN_RULES:
        return preferred_domain

    combined = f"{title} {body}".lower()
    scores: dict[str, int] = {}
    for domain, rule in DOMAIN_RULES.items():
        score = sum(1 for keyword in rule.verifiable_keywords if keyword.lower() in combined)
        scores[domain] = score
    return max(scores, key=scores.get) if any(scores.values()) else "society"


def classify_claim_type(sentence: str) -> str:
    lowered = sentence.lower()
    for claim_type, keywords in CLAIM_TYPE_KEYWORDS.items():
        if any(keyword.lower() in lowered for keyword in keywords):
            return claim_type
    return "descriptive_claim"


def is_verifiable(sentence: str, domain: str, claim_type: str) -> tuple[bool, str]:
    rule = DOMAIN_RULES[domain]
    if any(keyword in sentence for keyword in rule.non_verifiable_keywords):
        return False, "익명 또는 불명확한 출처 의존 표현이 있어 자동 검증 우선순위를 낮춥니다."
    if any(keyword in sentence for keyword in rule.verifiable_keywords):
        return True, "도메인별 검증 가능한 키워드가 포함되어 있습니다."
    if claim_type in {"numeric_stat", "date_event"} and (find_numbers(sentence) or find_dates(sentence)):
        return True, "수치 또는 날짜 기반 주장으로 외부 근거 대조가 가능합니다."
    if any(keyword in sentence for keyword in rule.weak_keywords):
        return False, "전망·해석성 표현 비중이 높아 강한 팩트체크 대상으로 보기 어렵습니다."
    return False, "현재 규칙 기준으로는 검증 가능성이 낮은 서술형 주장입니다."


def extract_claims(title: str, body: str, domain: str, context_date: str = "") -> list[Claim]:
    claims: list[Claim] = []
    sentences = split_sentences(body)
    for index, sentence in enumerate(sentences):
        if len(sentence) < 12:
            continue
        claim_type = classify_claim_type(sentence)
        verifiable, rationale = is_verifiable(sentence, domain=domain, claim_type=claim_type)
        claims.append(
            Claim(
                text=sentence,
                sentence_index=index,
                domain=domain,
                claim_type=claim_type,
                verifiable=verifiable,
                rationale=rationale,
                context_date=context_date,
                numbers=find_numbers(sentence),
                dates=find_dates(sentence),
                entities=find_entities(sentence),
            )
        )
    return claims


def build_summary(domain: str, claims: list[Claim], issues: list) -> dict[str, object]:
    verdict_counter = Counter(issue.verdict for issue in issues if issue.check_type == "external_fact")
    internal_counter = Counter(issue.label for issue in issues if issue.check_type == "internal_quality")
    return {
        "domain": domain,
        "total_claims": len(claims),
        "verifiable_claims": sum(1 for claim in claims if claim.verifiable),
        "non_verifiable_claims": sum(1 for claim in claims if not claim.verifiable),
        "external_verdicts": dict(verdict_counter),
        "internal_quality_flags": dict(internal_counter),
    }


def analyze_article(
    title: str,
    body: str,
    domain: str | None = None,
    evidence_docs: list[EvidenceDocument] | None = None,
    context_date: str = "",
) -> ArticleAnalysis:
    selected_domain = infer_domain(title=title, body=body, preferred_domain=domain)
    sentences = split_sentences(body)
    claims = extract_claims(title=title, body=body, domain=selected_domain, context_date=context_date)
    issues = review_internal_quality(claims=claims, sentences=sentences)
    if evidence_docs:
        issues.extend(verify_claims_against_evidence(claims=claims, evidence_docs=evidence_docs))
    summary = build_summary(domain=selected_domain, claims=claims, issues=issues)
    return ArticleAnalysis(
        domain=selected_domain,
        title=title,
        claim_count=len(claims),
        claims=claims,
        issues=issues,
        summary=summary,
    )
