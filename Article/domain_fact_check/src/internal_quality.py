from __future__ import annotations

from .models import AnalysisIssue, Claim
from .utils import lexical_overlap


GENERALIZATION_HINTS = ("전문가들은", "업계는", "누리꾼들은", "시민들은", "소비자들은", "연구진은")
WEAK_EVIDENCE_HINTS = ("가능성", "추정", "분석된다", "보인다", "해석된다", "전망된다")
STRONG_CONCLUSION_HINTS = ("입증", "증명", "확실", "반드시", "완전히", "결정적", "단정")
SINGLE_SOURCE_HINTS = ("한 연구", "한 관계자", "한 사례", "한 업체", "일부 사례")
CAUSAL_HINTS = ("때문", "영향", "원인", "결과", "효과", "유발", "초래")


def review_internal_quality(claims: list[Claim], sentences: list[str]) -> list[AnalysisIssue]:
    issues: list[AnalysisIssue] = []
    for claim in claims:
        sentence = claim.text
        lower_sentence = sentence.lower()

        if any(hint in sentence for hint in GENERALIZATION_HINTS) and claim.claim_type in {"causal_claim", "quote_or_attribution"}:
            issues.append(
                AnalysisIssue(
                    check_type="internal_quality",
                    label="overgeneralization_risk",
                    severity="medium",
                    sentence_index=claim.sentence_index,
                    claim_text=sentence,
                    reason="단일 또는 모호한 주체를 전체 집단 결론으로 확장하는 표현이 보입니다.",
                )
            )

        if any(hint in sentence for hint in STRONG_CONCLUSION_HINTS) and any(hint in sentence for hint in WEAK_EVIDENCE_HINTS):
            issues.append(
                AnalysisIssue(
                    check_type="internal_quality",
                    label="overstated_conclusion",
                    severity="high",
                    sentence_index=claim.sentence_index,
                    claim_text=sentence,
                    reason="근거 표현은 약한데 결론 표현은 강해서 주장-근거 강도가 맞지 않습니다.",
                )
            )

        if any(hint in sentence for hint in SINGLE_SOURCE_HINTS) and any(hint in sentence for hint in GENERALIZATION_HINTS):
            issues.append(
                AnalysisIssue(
                    check_type="internal_quality",
                    label="single_source_generalization",
                    severity="medium",
                    sentence_index=claim.sentence_index,
                    claim_text=sentence,
                    reason="단일 사례나 단일 출처를 일반 경향처럼 확장할 가능성이 있습니다.",
                )
            )

        if claim.claim_type == "causal_claim" and any(hint in lower_sentence for hint in CAUSAL_HINTS):
            previous_sentence = sentences[claim.sentence_index - 1] if claim.sentence_index > 0 else ""
            if previous_sentence and lexical_overlap(sentence, previous_sentence) < 0.2:
                issues.append(
                    AnalysisIssue(
                        check_type="internal_quality",
                        label="causal_leap_risk",
                        severity="medium",
                        sentence_index=claim.sentence_index,
                        claim_text=sentence,
                        reason="직전 문장과 연결 근거가 약한데 인과 표현을 사용하고 있습니다.",
                    )
                )

        if claim.claim_type == "numeric_stat" and len(claim.numbers) >= 2:
            issues.append(
                AnalysisIssue(
                    check_type="internal_quality",
                    label="multi_number_consistency_check",
                    severity="low",
                    sentence_index=claim.sentence_index,
                    claim_text=sentence,
                    reason="하나의 문장에 여러 수치가 있어 추가 대조가 필요합니다.",
                )
            )

    return issues
