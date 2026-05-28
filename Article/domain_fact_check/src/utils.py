from __future__ import annotations

import json
import re
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable


SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?다요])\s+|\n+")
NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)?(?:억|만|조|천)?(?:원|명|건|회|배|포인트|달러|%p|bp|%)?")
DATE_RE = re.compile(
    r"\d{4}년(?:\s*\d{1,2}월(?:\s*\d{1,2}일)?)?|\d{1,2}월\s*\d{1,2}일|\d{4}-\d{2}-\d{2}|"
    r"전날|이날|당일|어제|오늘|내일|지난\s*\d{1,2}일"
)
ENTITY_RE = re.compile(r"[A-Z][A-Za-z0-9&.-]{1,}|[가-힣]{2,}(?:청|부|원|원장|은행|증권|연구소|연구원|협회|공사|구단|팀|정부|법원)")
ALIAS_MAP = {
    "청와대": "대통령실",
    "대통령실": "대통령실",
    "한국은행": "한국은행",
    "bok": "한국은행",
    "금감원": "금융감독원",
    "금융감독원": "금융감독원",
    "통계청": "통계청",
    "검찰": "검찰",
    "검찰청": "검찰",
    "경찰": "경찰",
    "경찰청": "경찰",
}
LARGE_NUMBER_MULTIPLIER = {"조": 1_0000_0000_0000, "억": 100_000_000, "만": 10_000, "천": 1_000}
UNIT_CATEGORY = {
    "%": "percent",
    "%p": "rate_delta",
    "bp": "rate_delta",
    "원": "currency",
    "달러": "currency",
    "명": "count",
    "건": "count",
    "회": "count",
    "배": "ratio",
    "포인트": "point",
}


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    return [sentence.strip() for sentence in SENTENCE_SPLIT_RE.split(text) if sentence.strip()]


def find_numbers(text: str) -> list[str]:
    results: list[str] = []
    for match in NUMBER_RE.finditer(text):
        token = match.group(0)
        next_char = text[match.end():match.end() + 1]
        if next_char and next_char.strip() in {"년", "월", "일", "시", "분"}:
            continue
        results.append(token)
    return results


def find_dates(text: str) -> list[str]:
    return DATE_RE.findall(text)


def find_entities(text: str) -> list[str]:
    seen: set[str] = set()
    entities: list[str] = []
    for match in ENTITY_RE.findall(text):
        if match not in seen:
            seen.add(match)
            entities.append(match)
    return entities


def _normalize_number_token(token: str) -> str:
    token = token.strip().replace(",", "")
    token = re.sub(r"\s+", "", token)
    return token


def normalize_number_tokens(tokens: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    normalized: list[str] = []
    for token in tokens:
        value = _normalize_number_token(token)
        if value and value not in seen:
            seen.add(value)
            normalized.append(value)
    return normalized


def _normalize_date_token(token: str) -> str:
    token = token.strip()
    compact = re.sub(r"\s+", "", token)
    iso_match = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", compact)
    if iso_match:
        year, month, day = iso_match.groups()
        return f"{int(year):04d}-{int(month):02d}-{int(day):02d}"

    full_match = re.fullmatch(r"(\d{4})년(?:(\d{1,2})월(?:(\d{1,2})일)?)?", compact)
    if full_match:
        year, month, day = full_match.groups()
        if month and day:
            return f"{int(year):04d}-{int(month):02d}-{int(day):02d}"
        if month:
            return f"{int(year):04d}-{int(month):02d}"
        return f"{int(year):04d}"

    month_day_match = re.fullmatch(r"(\d{1,2})월(\d{1,2})일", compact)
    if month_day_match:
        month, day = month_day_match.groups()
        return f"{int(month):02d}-{int(day):02d}"

    return compact


def _parse_reference_date(reference_date: str | None) -> date | None:
    if not reference_date:
        return None
    for fmt in ("%Y-%m-%d", "%Y-%m", "%Y"):
        try:
            parsed = datetime.strptime(reference_date, fmt)
            if fmt == "%Y":
                return date(parsed.year, 1, 1)
            if fmt == "%Y-%m":
                return date(parsed.year, parsed.month, 1)
            return parsed.date()
        except ValueError:
            continue
    return None


def _resolve_relative_date(token: str, reference_date: str | None) -> str:
    ref = _parse_reference_date(reference_date)
    compact = re.sub(r"\s+", "", token)
    if ref is None:
        return compact
    if compact in {"이날", "당일", "오늘"}:
        return ref.isoformat()
    if compact in {"전날", "어제"}:
        return (ref - timedelta(days=1)).isoformat()
    if compact == "내일":
        return (ref + timedelta(days=1)).isoformat()
    relative_match = re.fullmatch(r"지난(\d{1,2})일", compact)
    if relative_match:
        day = int(relative_match.group(1))
        year = ref.year
        month = ref.month
        if day > ref.day:
            month -= 1
            if month == 0:
                year -= 1
                month = 12
        return f"{year:04d}-{month:02d}-{day:02d}"
    return compact


def normalize_date_tokens(tokens: Iterable[str], reference_date: str | None = None) -> list[str]:
    seen: set[str] = set()
    normalized: list[str] = []
    for token in tokens:
        relative_resolved = _resolve_relative_date(token, reference_date)
        value = _normalize_date_token(relative_resolved)
        if value and value not in seen:
            seen.add(value)
            normalized.append(value)
    return normalized


def sentence_has_entity_match(claim_entities: Iterable[str], sentence: str) -> bool:
    lowered = normalize_text(sentence)
    sentence_aliases = normalize_alias_tokens(find_entities(sentence))
    for entity in claim_entities:
        if not entity:
            continue
        normalized_entity = normalize_alias(entity)
        if normalize_text(entity) in lowered or normalized_entity in sentence_aliases:
            return True
    return False


def normalize_alias(entity: str) -> str:
    lowered = normalize_text(entity)
    return ALIAS_MAP.get(lowered, lowered)


def normalize_alias_tokens(tokens: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    normalized: list[str] = []
    for token in tokens:
        value = normalize_alias(token)
        if value and value not in seen:
            seen.add(value)
            normalized.append(value)
    return normalized


def _split_number_unit(token: str) -> tuple[str, str]:
    token = _normalize_number_token(token)
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)([%가-힣A-Za-z]+)?", token)
    if not match:
        return token, ""
    return match.group(1), match.group(2) or ""


def number_tokens_compatible(left: str, right: str) -> bool:
    left_value, left_unit = parse_numeric_token(left)
    right_value, right_unit = parse_numeric_token(right)
    if left_value is None or right_value is None:
        left_raw, left_raw_unit = _split_number_unit(left)
        right_raw, right_raw_unit = _split_number_unit(right)
        if left_raw != right_raw:
            return False
        if left_raw_unit == right_raw_unit:
            return True
        return not left_raw_unit or not right_raw_unit
    if abs(left_value - right_value) > 1e-9:
        return False
    if left_unit == right_unit:
        return True
    if not left_unit or not right_unit:
        return True
    return False


def date_tokens_compatible(left: str, right: str) -> bool:
    left_norm = _normalize_date_token(left)
    right_norm = _normalize_date_token(right)
    if left_norm == right_norm:
        return True
    return left_norm.startswith(right_norm) or right_norm.startswith(left_norm)


def count_compatible_tokens(claim_tokens: Iterable[str], evidence_tokens: Iterable[str], *, token_type: str) -> int:
    claim_list = list(claim_tokens)
    evidence_list = list(evidence_tokens)
    count = 0
    for claim_token in claim_list:
        for evidence_token in evidence_list:
            if token_type == "number" and number_tokens_compatible(claim_token, evidence_token):
                count += 1
                break
            if token_type == "date" and date_tokens_compatible(claim_token, evidence_token):
                count += 1
                break
    return count


def parse_numeric_token(token: str) -> tuple[float | None, str]:
    token = _normalize_number_token(token)
    token = token.replace("％", "%")
    if token.endswith("%p"):
        raw = token[:-2]
        try:
            return float(raw), UNIT_CATEGORY["%p"]
        except ValueError:
            return None, ""
    if token.endswith("bp"):
        raw = token[:-2]
        try:
            return float(raw) / 100.0, UNIT_CATEGORY["bp"]
        except ValueError:
            return None, ""

    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)(조|억|만|천)?([가-힣A-Za-z%]+)?", token)
    if not match:
        return None, ""
    raw_value, large_unit, trailing_unit = match.groups()
    try:
        value = float(raw_value)
    except ValueError:
        return None, ""

    if large_unit:
        value *= LARGE_NUMBER_MULTIPLIER[large_unit]

    trailing_unit = trailing_unit or ""
    unit_category = UNIT_CATEGORY.get(trailing_unit, trailing_unit)
    if trailing_unit == "%" and unit_category == "percent":
        return value, unit_category
    return value, unit_category


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def lexical_overlap(left: str, right: str) -> float:
    left_tokens = set(token for token in normalize_text(left).split() if len(token) > 1)
    right_tokens = set(token for token in normalize_text(right).split() if len(token) > 1)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens)


def load_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def load_json(path: str) -> object:
    return json.loads(load_text(path))
