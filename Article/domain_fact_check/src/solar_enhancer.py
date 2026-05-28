from __future__ import annotations

import os
from dataclasses import dataclass

from .config import load_project_dotenv
from .llm_adapter import LLMEnhancer
from .models import AnalysisIssue, ArticleAnalysis, Claim


DEFAULT_SOLAR_MODEL = "solar-mini"


@dataclass
class SolarLLMEnhancer(LLMEnhancer):
    """
    Upstage Solar enhancer skeleton.

    This class intentionally provides the same interface as OpenAILLMEnhancer,
    but leaves the provider-specific API call details for later implementation.
    Fill in the three public methods once the exact Solar API request/response
    contract is fixed in the backend environment.
    """

    model: str = DEFAULT_SOLAR_MODEL
    api_key: str | None = None
    base_url: str | None = None

    def __post_init__(self) -> None:
        load_project_dotenv()
        self.api_key = self.api_key or os.getenv("UPSTAGE_API_KEY")
        self.base_url = self.base_url or os.getenv("UPSTAGE_BASE_URL", "")

    def _not_implemented(self, method_name: str) -> RuntimeError:
        return RuntimeError(
            f"SolarLLMEnhancer.{method_name} is a provider skeleton only. "
            "Implement the actual Upstage Solar API call in this method before production use."
        )

    def extract_claims(self, *, title: str, body: str, domain: str, context_date: str = "") -> list[Claim]:
        raise self._not_implemented("extract_claims")

    def review_internal_quality(self, *, claims: list[Claim], body: str, domain: str) -> list[AnalysisIssue]:
        raise self._not_implemented("review_internal_quality")

    def build_user_report(self, analysis: ArticleAnalysis) -> dict[str, object]:
        raise self._not_implemented("build_user_report")
