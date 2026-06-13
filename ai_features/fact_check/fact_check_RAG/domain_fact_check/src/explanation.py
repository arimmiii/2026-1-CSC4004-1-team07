from __future__ import annotations

from collections import Counter

from .models import ArticleAnalysis, AnalysisIssue


VERDICT_LABELS = {
    "supported": "근거 일치",
    "contradicted": "반박됨",
    "misleading": "왜곡 가능성",
    "unverified": "근거 부족",
    "not_applicable": "해당 없음",
}


def _issue_line(issue: AnalysisIssue) -> str:
    prefix = f"[{issue.check_type}] {issue.label}"
    verdict = VERDICT_LABELS.get(issue.verdict, issue.verdict)
    if issue.evidence:
        evidence_title = issue.evidence[0].get("title", "")
        claim_text = issue.claim_text.strip()
        evidence_text = issue.evidence[0].get("text", "").strip()
        return (
            f"- {prefix}: {issue.reason} ({verdict}, 근거: {evidence_title})\n"
            f"  기사 문장: {claim_text}\n"
            f"  근거 문장: {evidence_text}"
        )
    return f"- {prefix}: {issue.reason} ({verdict})"


def build_report_payload(analysis: ArticleAnalysis) -> dict[str, object]:
    external_issues = [issue for issue in analysis.issues if issue.check_type == "external_fact"]
    internal_issues = [issue for issue in analysis.issues if issue.check_type == "internal_quality"]
    verdicts = Counter(issue.verdict for issue in external_issues)

    overall = "검증 가능 claim이 충분하지 않습니다."
    if verdicts.get("contradicted"):
        overall = "외부 근거와 충돌하는 claim이 있습니다."
    elif verdicts.get("misleading"):
        overall = "직접 반박까지는 아니지만 왜곡 가능성이 있는 claim이 있습니다."
    elif verdicts.get("supported") and not verdicts.get("unverified"):
        overall = "대조된 claim은 대체로 외부 근거와 일치합니다."

    return {
        "title": analysis.title,
        "domain": analysis.domain,
        "overall_assessment": overall,
        "claim_count": analysis.claim_count,
        "verifiable_claim_count": analysis.summary["verifiable_claims"],
        "external_verdicts": dict(verdicts),
        "internal_issue_count": len(internal_issues),
        "top_external_issues": [_issue_line(issue) for issue in external_issues[:5]],
        "top_internal_issues": [_issue_line(issue) for issue in internal_issues[:5]],
        "detailed_external_issues": [
            {
                "label": issue.label,
                "verdict": issue.verdict,
                "severity": issue.severity,
                "claim_text": issue.claim_text,
                "reason": issue.reason,
                "evidence_title": issue.evidence[0].get("title", "") if issue.evidence else "",
                "evidence_text": issue.evidence[0].get("text", "") if issue.evidence else "",
                "evidence_url": issue.evidence[0].get("url", "") if issue.evidence else "",
            }
            for issue in external_issues[:10]
        ],
    }


def build_report(analysis: ArticleAnalysis) -> str:
    payload = build_report_payload(analysis)
    lines = [
        f"제목: {payload['title']}",
        f"분야: {payload['domain']}",
        f"총 claim 수: {payload['claim_count']}",
        f"검증 가능 claim 수: {payload['verifiable_claim_count']}",
        f"총평: {payload['overall_assessment']}",
    ]

    top_external = payload["top_external_issues"]
    if top_external:
        lines.append("")
        lines.append("외부 근거 비교:")
        lines.extend(top_external)

    top_internal = payload["top_internal_issues"]
    if top_internal:
        lines.append("")
        lines.append("내부 품질 검토:")
        lines.extend(top_internal)

    return "\n".join(lines)
