from __future__ import annotations

from .models import AnalysisIssue, Claim, EvidenceDocument
from .utils import (
    count_compatible_tokens,
    find_dates,
    find_numbers,
    lexical_overlap,
    normalize_date_tokens,
    normalize_number_tokens,
    sentence_has_entity_match,
    split_sentences,
)


def _best_match_score(claim: Claim, sentence: str) -> float:
    score = lexical_overlap(claim.text, sentence)

    claim_numbers = normalize_number_tokens(claim.numbers)
    sentence_numbers = normalize_number_tokens(find_numbers(sentence))
    if claim_numbers and sentence_numbers:
        overlap = count_compatible_tokens(claim_numbers, sentence_numbers, token_type="number")
        score += 0.12 * overlap

    claim_dates = normalize_date_tokens(claim.dates)
    sentence_dates = normalize_date_tokens(find_dates(sentence))
    if claim_dates and sentence_dates:
        overlap = count_compatible_tokens(claim_dates, sentence_dates, token_type="date")
        score += 0.15 * overlap

    if claim.entities and sentence_has_entity_match(claim.entities, sentence):
        score += 0.1

    return score


def _detailed_reason(
    *,
    verdict: str,
    claim: Claim,
    sentence: str,
    claim_numbers: list[str],
    sentence_numbers: list[str],
    claim_dates: list[str],
    sentence_dates: list[str],
) -> str:
    if verdict == "contradicted":
        if claim_numbers and sentence_numbers and set(claim_numbers) != set(sentence_numbers):
            return (
                f"기사 문장은 {claim_numbers}라고 주장하지만, "
                f"근거 문장은 {sentence_numbers}로 확인됩니다. "
                "같은 주장으로 보이는 문맥에서 수치가 다릅니다."
            )
        if claim_dates and sentence_dates and set(claim_dates) != set(sentence_dates):
            return (
                f"기사 문장은 시점을 {claim_dates}로 적었지만, "
                f"근거 문장은 {sentence_dates}로 확인됩니다. "
                "같은 주장으로 보이는 문맥에서 시점이 다릅니다."
            )
        return (
            "기사 문장과 근거 문장이 같은 사안을 설명하는 것으로 보이지만, "
            "핵심 사실이 서로 충돌합니다."
        )

    if verdict == "misleading":
        if claim_numbers and sentence_numbers:
            return (
                f"기사 문장은 {claim_numbers}를 포함하지만, "
                f"근거 문장은 {sentence_numbers}로 일부만 맞습니다. "
                "핵심 수치가 완전히 일치하지 않아 표현이 과하거나 일부만 맞을 가능성이 있습니다."
            )
        if claim_dates and sentence_dates:
            return (
                f"기사 문장은 시점을 {claim_dates}로 적었지만, "
                f"근거 문장은 {sentence_dates}로 일부만 맞습니다. "
                "시점 설명이 단순화되거나 뭉뚱그려졌을 가능성이 있습니다."
            )
        return (
            "핵심 키워드는 겹치지만, 근거 문장만으로는 기사 문장의 결론 강도까지 "
            "직접 뒷받침되지 않습니다."
        )

    if verdict == "unverified":
        return (
            "검색된 근거 문장만으로는 기사 문장을 직접 지지하거나 반박하기 어렵습니다. "
            "수치나 날짜, 사건 설명이 충분히 대응되지 않습니다."
        )

    return (
        "기사 문장의 핵심 요소와 근거 문장의 수치·시점·사건 설명이 대체로 일치합니다."
    )


def verify_claims_against_evidence(claims: list[Claim], evidence_docs: list[EvidenceDocument]) -> list[AnalysisIssue]:
    issues: list[AnalysisIssue] = []
    if not evidence_docs:
        return issues

    evidence_sentences: list[tuple[EvidenceDocument, str]] = []
    for doc in evidence_docs:
        for sentence in split_sentences(doc.text):
            evidence_sentences.append((doc, sentence))

    for claim in claims:
        if not claim.verifiable:
            continue

        best_match = None
        best_score = 0.0
        for doc, sentence in evidence_sentences:
            score = _best_match_score(claim, sentence)
            if score > best_score:
                best_match = (doc, sentence)
                best_score = score

        if best_match is None or best_score < 0.25:
            issues.append(
                AnalysisIssue(
                    check_type="external_fact",
                    label="insufficient_evidence",
                    severity="medium",
                    sentence_index=claim.sentence_index,
                    claim_text=claim.text,
                    reason="제공된 증거 문서에서 직접 대응되는 근거 문장을 찾지 못했습니다.",
                    verdict="unverified",
                )
            )
            continue

        doc, sentence = best_match
        claim_numbers = normalize_number_tokens(claim.numbers)
        claim_dates = normalize_date_tokens(claim.dates, reference_date=claim.context_date or None)
        sentence_numbers = normalize_number_tokens(find_numbers(sentence))
        sentence_dates = normalize_date_tokens(find_dates(sentence), reference_date=doc.published_at or claim.context_date or None)
        matched_number_count = count_compatible_tokens(claim_numbers, sentence_numbers, token_type="number")
        matched_date_count = count_compatible_tokens(claim_dates, sentence_dates, token_type="date")

        verdict = "supported"
        if claim_numbers and sentence_numbers:
            if matched_number_count != len(claim_numbers):
                verdict = "contradicted" if matched_number_count == 0 else "misleading"
        elif claim_numbers and not sentence_numbers:
            verdict = "unverified"

        if verdict == "supported" and claim_dates and sentence_dates:
            if matched_date_count != len(claim_dates):
                verdict = "contradicted" if matched_date_count == 0 else "misleading"
        elif verdict == "supported" and claim_dates and not sentence_dates:
            verdict = "unverified"

        if verdict == "supported" and claim.claim_type == "causal_claim" and best_score < 0.45:
            verdict = "misleading"
        elif verdict == "supported" and best_score < 0.35:
            verdict = "unverified"

        if verdict == "supported" and claim.entities and not sentence_has_entity_match(claim.entities, sentence):
            verdict = "unverified"

        reason = _detailed_reason(
            verdict=verdict,
            claim=claim,
            sentence=sentence,
            claim_numbers=claim_numbers,
            sentence_numbers=sentence_numbers,
            claim_dates=claim_dates,
            sentence_dates=sentence_dates,
        )

        issues.append(
            AnalysisIssue(
                check_type="external_fact",
                label=f"claim_{verdict}",
                severity="high" if verdict == "contradicted" else "medium",
                sentence_index=claim.sentence_index,
                claim_text=claim.text,
                reason=reason,
                verdict=verdict,
                evidence=[
                    {
                        "title": doc.title,
                        "url": doc.url,
                        "text": sentence,
                    }
                ],
            )
        )

    return issues
