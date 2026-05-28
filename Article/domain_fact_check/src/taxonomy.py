from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DomainRule:
    name: str
    verifiable_keywords: tuple[str, ...]
    weak_keywords: tuple[str, ...]
    non_verifiable_keywords: tuple[str, ...]
    trusted_source_hints: tuple[str, ...]


DOMAIN_RULES: dict[str, DomainRule] = {
    "economy": DomainRule(
        name="economy",
        verifiable_keywords=(
            "금리",
            "환율",
            "물가",
            "실업률",
            "수출",
            "수입",
            "매출",
            "영업이익",
            "주가",
            "예산",
            "세수",
            "공시",
            "gdp",
        ),
        weak_keywords=("전망", "우려", "심리", "예상", "가능성"),
        non_verifiable_keywords=("익명", "관계자", "카더라", "소문"),
        trusted_source_hints=("통계청", "한국은행", "금융감독원", "dart", "기획재정부", "산업통상자원부"),
    ),
    "society": DomainRule(
        name="society",
        verifiable_keywords=(
            "경찰",
            "소방",
            "정부",
            "지자체",
            "통계",
            "브리핑",
            "공개",
            "발표",
            "확인",
            "집계",
        ),
        weak_keywords=("추정", "정황", "제보", "목격", "논란"),
        non_verifiable_keywords=("익명", "관계자", "제보자", "카더라"),
        trusted_source_hints=("경찰청", "소방청", "행정안전부", "질병관리청", "서울시", "보건복지부"),
    ),
    "science": DomainRule(
        name="science",
        verifiable_keywords=(
            "논문",
            "연구",
            "실험",
            "표본",
            "저널",
            "학회",
            "발표",
            "데이터",
            "분석",
            "통계적",
            "유의",
        ),
        weak_keywords=("가능성", "기대", "획기적", "혁신", "추정"),
        non_verifiable_keywords=("익명", "관계자"),
        trusted_source_hints=("nature", "science", "cell", "pubmed", "arxiv", "kist", "kaist", "서울대"),
    ),
    "lifestyle_culture": DomainRule(
        name="lifestyle_culture",
        verifiable_keywords=(
            "개봉",
            "출간",
            "공연",
            "전시",
            "축제",
            "공식",
            "발표",
            "출시",
            "일정",
            "수상",
            "식약처",
        ),
        weak_keywords=("인기", "화제", "논란", "감성", "호평"),
        non_verifiable_keywords=("익명", "관계자", "업계에 따르면"),
        trusted_source_hints=("영화진흥위원회", "예스24", "교보문고", "식품의약품안전처", "문화체육관광부"),
    ),
}


CLAIM_TYPE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "numeric_stat": ("%", "억원", "조원", "명", "건", "배", "포인트", "달러", "원"),
    "date_event": ("년", "월", "일", "개최", "시행", "발표", "개봉", "출간"),
    "quote_or_attribution": ("라고", "밝혔다", "설명했다", "주장했다", "말했다", "전했다"),
    "causal_claim": ("때문", "영향", "원인", "결과", "입증", "증명", "효과"),
}
