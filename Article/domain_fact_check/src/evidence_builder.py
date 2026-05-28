from __future__ import annotations

from urllib.parse import urlparse

from .models import Claim, EvidenceDocument
from .taxonomy import DOMAIN_RULES
from .utils import normalize_text


SEARCH_TITLE_KEYS = ("title", "name", "headline")
SEARCH_URL_KEYS = ("url", "link")
SEARCH_TEXT_KEYS = ("text", "content", "body", "snippet", "description")
SEARCH_SOURCE_KEYS = ("source_type", "source", "site_name", "publisher")
SEARCH_DATE_KEYS = ("published_at", "date", "published", "pubDate")


def canonicalize_url(url: str) -> str:
    if not url:
        return ""
    parsed = urlparse(url.strip())
    netloc = parsed.netloc.lower()
    path = parsed.path.rstrip("/")
    return f"{netloc}{path}"


def _title_core(title: str) -> str:
    if not title:
        return ""
    core = title.strip()
    if " - " in core:
        core = core.rsplit(" - ", 1)[0].strip()
    return normalize_text(core)


def _title_source_hint(title: str) -> str:
    if not title:
        return ""
    if " - " in title:
        return normalize_text(title.rsplit(" - ", 1)[-1].strip())
    return ""


def _title_overlap_score(left: str, right: str) -> float:
    left_core = _title_core(left)
    right_core = _title_core(right)
    if not left_core or not right_core:
        return 0.0
    left_tokens = set(token for token in left_core.split() if len(token) > 1)
    right_tokens = set(token for token in right_core.split() if len(token) > 1)
    if not left_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens)


def _same_article_by_title_and_source(*, article_title: str, candidate_title: str, candidate_url: str, candidate_source_name: str) -> bool:
    article_core = _title_core(article_title)
    candidate_core = _title_core(candidate_title)
    if not article_core or not candidate_core:
        return False

    article_tokens = set(token for token in article_core.split() if len(token) > 1)
    candidate_tokens = set(token for token in candidate_core.split() if len(token) > 1)
    overlap_score = 0.0
    if article_tokens:
        overlap_score = len(article_tokens & candidate_tokens) / len(article_tokens)

    same_core = (
        article_core == candidate_core
        or article_core in candidate_core
        or candidate_core in article_core
        or overlap_score >= 0.6
    )
    if not same_core:
        return False

    article_source_hint = _title_source_hint(article_title)
    candidate_source_text = normalize_text(f"{candidate_title} {candidate_source_name} {urlparse(candidate_url).netloc}")
    if article_source_hint and article_source_hint in candidate_source_text:
        return True
    return False


def _pick_first(payload: dict, keys: tuple[str, ...], default: str = "") -> str:
    for key in keys:
        value = payload.get(key, "")
        if value:
            return str(value).strip()
    return default


def _infer_source_type(url: str, source_name: str, domain: str) -> str:
    host = urlparse(url).netloc.lower()
    rule = DOMAIN_RULES.get(domain)
    hints = tuple(hint.lower() for hint in rule.trusted_source_hints) if rule else ()
    if any(hint in source_name.lower() for hint in hints) or any(hint in host for hint in hints):
        return "trusted_source"
    if "gov" in host or "go.kr" in host or "ac.kr" in host or "edu" in host:
        return "official_site"
    if host:
        return "news_or_web"
    return "unknown"


def score_evidence_document(doc: EvidenceDocument, domain: str, claims: list[Claim] | None = None) -> float:
    score = 0.0
    if doc.source_type in {"trusted_source", "official_site"}:
        score += 3.0
    if doc.published_at:
        score += 0.5

    normalized_text = normalize_text(f"{doc.title} {doc.text}")
    rule = DOMAIN_RULES[domain]
    for keyword in rule.verifiable_keywords:
        if keyword.lower() in normalized_text:
            score += 0.3

    for hint in rule.trusted_source_hints:
        if hint.lower() in normalized_text or hint.lower() in doc.url.lower():
            score += 0.8

    if claims:
        for claim in claims:
            for token in claim.entities + claim.numbers + claim.dates:
                if token and token.lower() in normalized_text:
                    score += 0.4
    return score


def build_evidence_documents(
    raw_results: list[dict],
    domain: str,
    claims: list[Claim] | None = None,
    top_k: int = 5,
    excluded_url: str = "",
    article_title: str = "",
) -> list[EvidenceDocument]:
    docs: list[EvidenceDocument] = []
    seen_keys: set[tuple[str, str]] = set()
    seen_title_families: list[str] = []
    excluded_key = canonicalize_url(excluded_url)

    for item in raw_results:
        title = _pick_first(item, SEARCH_TITLE_KEYS)
        url = _pick_first(item, SEARCH_URL_KEYS)
        text = _pick_first(item, SEARCH_TEXT_KEYS)
        source_name = _pick_first(item, SEARCH_SOURCE_KEYS)
        published_at = _pick_first(item, SEARCH_DATE_KEYS)
        if not text and not title:
            continue
        if excluded_key and canonicalize_url(url) == excluded_key:
            continue
        if article_title and _same_article_by_title_and_source(
            article_title=article_title,
            candidate_title=title,
            candidate_url=url,
            candidate_source_name=source_name,
        ):
            continue
        title_core = _title_core(title)
        if title_core and any(_title_overlap_score(title_core, seen_title) >= 0.75 for seen_title in seen_title_families):
            continue

        dedupe_key = (url, title)
        if dedupe_key in seen_keys:
            continue
        seen_keys.add(dedupe_key)
        if title_core:
            seen_title_families.append(title_core)

        docs.append(
            EvidenceDocument(
                title=title or source_name or "untitled",
                text=text,
                url=url,
                source_type=_infer_source_type(url=url, source_name=source_name, domain=domain),
                domain=domain,
                published_at=published_at,
            )
        )

    ranked_docs = sorted(docs, key=lambda doc: score_evidence_document(doc, domain=domain, claims=claims), reverse=True)
    return ranked_docs[:top_k]


def generate_search_queries(title: str, claims: list[Claim], domain: str, limit: int = 8) -> list[str]:
    queries: list[str] = []
    rule = DOMAIN_RULES[domain]

    if title.strip():
        queries.append(title.strip())

    for claim in claims:
        if not claim.verifiable:
            continue
        parts = [claim.text]
        if claim.entities:
            parts.append(" ".join(claim.entities[:3]))
        if claim.numbers:
            parts.append(" ".join(claim.numbers[:2]))
        if claim.dates:
            parts.append(" ".join(claim.dates[:1]))
        query = " ".join(part for part in parts if part).strip()
        if query:
            queries.append(query)

    for hint in rule.trusted_source_hints[:3]:
        queries.append(f"{title.strip()} {hint}".strip())

    deduped: list[str] = []
    seen: set[str] = set()
    for query in queries:
        normalized = normalize_text(query)
        if normalized and normalized not in seen:
            seen.add(normalized)
            deduped.append(query)
    return deduped[:limit]
