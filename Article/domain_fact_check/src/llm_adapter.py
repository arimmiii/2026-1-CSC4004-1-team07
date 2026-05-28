from __future__ import annotations

from typing import Protocol

from .models import AnalysisIssue, ArticleAnalysis, Claim


class LLMEnhancer(Protocol):
    def extract_claims(self, *, title: str, body: str, domain: str, context_date: str = "") -> list[Claim]:
        ...

    def review_internal_quality(self, *, claims: list[Claim], body: str, domain: str) -> list[AnalysisIssue]:
        ...

    def build_user_report(self, analysis: ArticleAnalysis) -> dict[str, object]:
        ...
