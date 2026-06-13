from __future__ import annotations

import os
from dataclasses import dataclass
import logging

from .config import load_project_dotenv
from .llm_schemas import LLMClaimExtraction, LLMInternalReview, LLMReport
from .logging_utils import log_json_event
from .models import AnalysisIssue, ArticleAnalysis, Claim
from .utils import find_dates, find_entities, find_numbers, split_sentences

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None


DEFAULT_LLM_MODEL = "gpt-5.4-mini"


@dataclass
class OpenAILLMEnhancer:
    model: str = DEFAULT_LLM_MODEL
    api_key: str | None = None
    reasoning_effort: str | None = None
    logger: logging.Logger | None = None

    def __post_init__(self) -> None:
        load_project_dotenv()
        self.api_key = self.api_key or os.getenv("OPENAI_API_KEY")
        if OpenAI is None:
            raise RuntimeError("openai package is not installed.")
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY environment variable is not set.")
        self.client = OpenAI(api_key=self.api_key)

    def _response_kwargs(self) -> dict[str, object]:
        kwargs: dict[str, object] = {"model": self.model}
        if self.reasoning_effort:
            kwargs["reasoning"] = {"effort": self.reasoning_effort}
        return kwargs

    def extract_claims(self, *, title: str, body: str, domain: str, context_date: str = "") -> list[Claim]:
        log_json_event(self.logger, "llm_extract_claims_start", {"domain": domain, "title": title, "body_chars": len(body), "context_date": context_date})
        response = self.client.responses.parse(
            **self._response_kwargs(),
            input=[
                {
                    "role": "system",
                    "content": (
                        "You extract factual claims from a Korean news article. "
                        "Return only claims that are meaningful for fact-checking in the given domain. "
                        "Mark claims as verifiable only when they can plausibly be checked with public evidence."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"domain: {domain}\n"
                        f"context_date: {context_date}\n"
                        f"title: {title}\n"
                        f"body: {body}\n"
                        "Return claim text, claim_type, verifiable, and a short rationale."
                    ),
                },
            ],
            text_format=LLMClaimExtraction,
        )
        parsed = response.output_parsed
        sentences = split_sentences(body)
        claims: list[Claim] = []
        for item in parsed.claims:
            sentence_index = next((idx for idx, sentence in enumerate(sentences) if item.text in sentence or sentence in item.text), -1)
            claims.append(
                Claim(
                    text=item.text,
                    sentence_index=max(sentence_index, 0),
                    domain=domain,
                    claim_type=item.claim_type,
                    verifiable=item.verifiable,
                    rationale=item.rationale,
                    context_date=context_date,
                    numbers=find_numbers(item.text),
                    dates=find_dates(item.text),
                    entities=find_entities(item.text),
                )
            )
        log_json_event(self.logger, "llm_extract_claims_done", {"domain": domain, "claim_count": len(claims)})
        return claims

    def review_internal_quality(self, *, claims: list[Claim], body: str, domain: str) -> list[AnalysisIssue]:
        log_json_event(self.logger, "llm_internal_review_start", {"domain": domain, "claim_count": len(claims), "body_chars": len(body)})
        response = self.client.responses.parse(
            **self._response_kwargs(),
            input=[
                {
                    "role": "system",
                    "content": (
                        "You review a news article for internal quality problems. "
                        "Focus on overclaiming, causal leaps, unsupported generalization, quote/context distortion, "
                        "and mismatch between evidence strength and conclusion strength."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"domain: {domain}\n"
                        f"body: {body}\n"
                        f"claims: {[claim.text for claim in claims]}\n"
                        "Return issue label, severity, claim_text, and reason."
                    ),
                },
            ],
            text_format=LLMInternalReview,
        )
        parsed = response.output_parsed
        claim_index = {claim.text: claim.sentence_index for claim in claims}
        issues: list[AnalysisIssue] = []
        for item in parsed.issues:
            issues.append(
                AnalysisIssue(
                    check_type="internal_quality",
                    label=item.label,
                    severity=item.severity,
                    sentence_index=claim_index.get(item.claim_text, 0),
                    claim_text=item.claim_text,
                    reason=item.reason,
                )
            )
        log_json_event(self.logger, "llm_internal_review_done", {"domain": domain, "issue_count": len(issues)})
        return issues

    def build_user_report(self, analysis: ArticleAnalysis) -> dict[str, object]:
        log_json_event(self.logger, "llm_user_report_start", {"domain": analysis.domain, "claim_count": analysis.claim_count, "issue_count": len(analysis.issues)})
        external_issue_context = []
        for issue in analysis.issues:
            if issue.check_type != "external_fact":
                continue
            evidence = issue.evidence[0] if issue.evidence else {}
            external_issue_context.append(
                {
                    "verdict": issue.verdict,
                    "claim_text": issue.claim_text,
                    "reason": issue.reason,
                    "evidence_title": evidence.get("title", ""),
                    "evidence_text": evidence.get("text", ""),
                }
            )
        response = self.client.responses.parse(
            **self._response_kwargs(),
            input=[
                {
                    "role": "system",
                    "content": (
                        "You summarize a fact-check analysis for end users in Korean. "
                        "Be factual, avoid speculation, and explain the result transparently. "
                        "When there is a problem, point to the exact article claim and the exact evidence sentence. "
                        "Explain why the evidence supports, contradicts, or fails to verify the claim."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"analysis: {analysis.to_dict()}\n"
                        f"external_issue_context: {external_issue_context}\n"
                        "Return an overall assessment, a short user summary, key bullet points, "
                        "and issue_explanations. Each issue explanation should explicitly mention "
                        "the exact article claim, the exact evidence sentence, and why that leads to the verdict. "
                        "Do not summarize vaguely as '근거가 부족하다' without naming which claim and which evidence."
                    ),
                },
            ],
            text_format=LLMReport,
        )
        parsed = response.output_parsed
        result = {
            "overall_assessment": parsed.overall_assessment,
            "user_summary": parsed.user_summary,
            "key_points": parsed.key_points,
            "issue_explanations": parsed.issue_explanations,
        }
        log_json_event(
            self.logger,
            "llm_user_report_done",
            {
                "domain": analysis.domain,
                "key_point_count": len(parsed.key_points),
                "issue_explanation_count": len(parsed.issue_explanations),
            },
        )
        return result
