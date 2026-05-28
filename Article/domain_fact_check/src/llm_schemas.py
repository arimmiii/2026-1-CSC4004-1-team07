from __future__ import annotations

from pydantic import BaseModel, Field


class LLMClaim(BaseModel):
    text: str
    claim_type: str = Field(default="descriptive_claim")
    verifiable: bool = Field(default=False)
    rationale: str = Field(default="")


class LLMClaimExtraction(BaseModel):
    claims: list[LLMClaim]


class LLMInternalIssue(BaseModel):
    label: str
    severity: str
    claim_text: str
    reason: str


class LLMInternalReview(BaseModel):
    issues: list[LLMInternalIssue]


class LLMReport(BaseModel):
    overall_assessment: str
    user_summary: str
    key_points: list[str]
    issue_explanations: list[str] = Field(default_factory=list)
