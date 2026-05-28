from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


SUPPORTED_DOMAINS = ("economy", "society", "science", "lifestyle_culture")
VERDICTS = ("supported", "contradicted", "misleading", "unverified", "not_applicable")
CHECK_TYPES = ("external_fact", "internal_quality")


@dataclass
class EvidenceDocument:
    title: str
    text: str
    url: str = ""
    source_type: str = ""
    domain: str = ""
    published_at: str = ""


@dataclass
class Claim:
    text: str
    sentence_index: int
    domain: str
    claim_type: str
    verifiable: bool
    rationale: str
    context_date: str = ""
    numbers: list[str] = field(default_factory=list)
    dates: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)


@dataclass
class AnalysisIssue:
    check_type: str
    label: str
    severity: str
    sentence_index: int
    claim_text: str
    reason: str
    evidence: list[dict[str, str]] = field(default_factory=list)
    verdict: str = "not_applicable"


@dataclass
class ArticleAnalysis:
    domain: str
    title: str
    claim_count: int
    claims: list[Claim]
    issues: list[AnalysisIssue]
    summary: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "domain": self.domain,
            "title": self.title,
            "claim_count": self.claim_count,
            "claims": [asdict(claim) for claim in self.claims],
            "issues": [asdict(issue) for issue in self.issues],
            "summary": self.summary,
        }
