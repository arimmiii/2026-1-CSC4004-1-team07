# Domain Fact Check Output JSON Spec

이 문서는 `domain_fact_check` 모듈이 반환하는 JSON 구조를 별도로 정리한 문서입니다.

기준 함수:

- `src/service.py`
  - `run_fact_check_service()`
  - `run_fact_check_with_evidence()`

---

## 1. 최상위 JSON 구조

기본 반환 형식:

```json
{
  "domain": "economy",
  "queries": [],
  "raw_search_results": [],
  "evidence_documents": [],
  "analysis": {},
  "report_payload": {},
  "report_text": "..."
}
```

### 1-1. `domain`

- 타입: `string`
- 의미: 최종 적용된 내부 도메인
- 가능한 값:
  - `economy`
  - `society`
  - `science`
  - `lifestyle_culture`

예:

```json
"domain": "science"
```

### 1-2. `queries`

- 타입: `string[]`
- 의미: 검색 API에 실제로 사용한 검색어 목록

예:

```json
"queries": [
  "정부, 4월 물가 상승률 2.3% 발표",
  "정부, 4월 물가 상승률 2.3% 발표 통계청"
]
```

### 1-3. `raw_search_results`

- 타입: `object[]`
- 의미: 검색엔진이 반환한 원본 결과
- 대표 필드:
  - `title`
  - `link` 또는 `url`
  - `snippet` 또는 `text`
  - `source`
  - `published_at`

예:

```json
"raw_search_results": [
  {
    "title": "통계청 보도자료",
    "link": "https://example.org/source",
    "snippet": "2026년 4월 소비자물가 상승률은 2.1%였다.",
    "source": "통계청",
    "published_at": "2026-05-01"
  }
]
```

### 1-4. `evidence_documents`

- 타입: `object[]`
- 의미: 검색 결과를 팩트체크용 evidence 구조로 정규화한 결과

형식:

```json
{
  "title": "string",
  "text": "string",
  "url": "string",
  "source_type": "string",
  "domain": "string",
  "published_at": "string"
}
```

예:

```json
"evidence_documents": [
  {
    "title": "통계청 보도자료",
    "text": "2026년 4월 소비자물가 상승률은 2.1%였다.",
    "url": "https://example.org/source",
    "source_type": "trusted_source",
    "domain": "economy",
    "published_at": "2026-05-01"
  }
]
```

#### `source_type` 가능한 값

- `trusted_source`
- `official_site`
- `news_or_web`
- `unknown`

---

## 2. `analysis` 구조

핵심 분석 결과입니다.

형식:

```json
"analysis": {
  "domain": "string",
  "title": "string",
  "claim_count": 0,
  "claims": [],
  "issues": [],
  "summary": {}
}
```

### 2-1. `analysis.domain`

- 타입: `string`
- 의미: 분석된 기사의 내부 도메인

### 2-2. `analysis.title`

- 타입: `string`
- 의미: 기사 제목

### 2-3. `analysis.claim_count`

- 타입: `number`
- 의미: 추출된 claim 개수

### 2-4. `analysis.claims`

- 타입: `object[]`
- 의미: 기사에서 추출한 claim 목록

형식:

```json
{
  "text": "string",
  "sentence_index": 0,
  "domain": "string",
  "claim_type": "string",
  "verifiable": true,
  "rationale": "string",
  "numbers": [],
  "dates": [],
  "entities": []
}
```

#### 필드 설명

- `text`
  - claim 문장 원문
- `sentence_index`
  - 본문에서 몇 번째 문장인지
- `domain`
  - claim에 적용된 도메인
- `claim_type`
  - 예:
    - `numeric_stat`
    - `date_event`
    - `quote_or_attribution`
    - `causal_claim`
    - `descriptive_claim`
- `verifiable`
  - `true` / `false`
  - 외부 근거로 검증 가능한 claim인지
- `rationale`
  - 왜 verifiable / non-verifiable로 봤는지 설명
- `numbers`
  - 문장에서 추출된 숫자들
- `dates`
  - 문장에서 추출된 날짜들
- `entities`
  - 문장에서 추출된 엔티티들

예:

```json
{
  "text": "정부는 2026년 4월 소비자물가 상승률이 2.3%라고 밝혔다.",
  "sentence_index": 0,
  "domain": "economy",
  "claim_type": "numeric_stat",
  "verifiable": true,
  "rationale": "도메인별 검증 가능한 키워드가 포함되어 있습니다.",
  "numbers": ["2026", "4", "2.3%"],
  "dates": ["2026년 4월"],
  "entities": []
}
```

### 2-5. `analysis.issues`

- 타입: `object[]`
- 의미: 실제 판정 결과 + 내부 품질 검토 결과

형식:

```json
{
  "check_type": "string",
  "label": "string",
  "severity": "string",
  "sentence_index": 0,
  "claim_text": "string",
  "reason": "string",
  "evidence": [],
  "verdict": "string"
}
```

#### `check_type`

가능한 값:

- `external_fact`
- `internal_quality`

#### A. 외부 팩트체크 결과

예:

```json
{
  "check_type": "external_fact",
  "label": "claim_contradicted",
  "severity": "high",
  "sentence_index": 0,
  "claim_text": "정부는 2026년 4월 소비자물가 상승률이 2.3%라고 밝혔다.",
  "reason": "주장 수치와 증거 수치가 일치하지 않습니다.",
  "evidence": [
    {
      "title": "통계청 보도자료",
      "url": "https://example.org/source",
      "text": "2026년 4월 소비자물가 상승률은 2.1%였다."
    }
  ],
  "verdict": "contradicted"
}
```

##### `verdict` 가능한 값

- `supported`
- `contradicted`
- `misleading`
- `unverified`

##### `severity`

보통:

- `high`
- `medium`
- `low`

##### `evidence`

- 타입: `object[]`
- claim 판단에 사용한 evidence 문장/문서

내부 구조:

```json
{
  "title": "string",
  "url": "string",
  "text": "string"
}
```

#### B. 내부 품질 검토 결과

예:

```json
{
  "check_type": "internal_quality",
  "label": "multi_number_consistency_check",
  "severity": "low",
  "sentence_index": 0,
  "claim_text": "정부는 2026년 4월 소비자물가 상승률이 2.3%라고 밝혔다.",
  "reason": "하나의 문장에 여러 수치가 있어 추가 대조가 필요합니다.",
  "evidence": [],
  "verdict": "not_applicable"
}
```

##### `label` 예시

- `multi_number_consistency_check`
- `overgeneralization_risk`
- `single_source_generalization`
- `causal_leap_risk`

LLM 보강이 붙으면 더 다양한 라벨이 나올 수 있습니다.
예:

- `Overclaiming`
- `Quote/Context Distortion`
- `Unsupported Generalization`

##### `verdict`

내부 품질 검토는 보통:

- `not_applicable`

### 2-6. `analysis.summary`

- 타입: `object`
- 의미: 기사 단위 집계 결과

형식:

```json
{
  "domain": "string",
  "total_claims": 0,
  "verifiable_claims": 0,
  "non_verifiable_claims": 0,
  "external_verdicts": {},
  "internal_quality_flags": {}
}
```

#### 필드 설명

- `domain`
  - 기사 도메인
- `total_claims`
  - claim 총 개수
- `verifiable_claims`
  - 검증 가능한 claim 개수
- `non_verifiable_claims`
  - 검증 불가 claim 개수
- `external_verdicts`
  - verdict별 개수 집계
- `internal_quality_flags`
  - 내부 품질 경고 라벨별 개수

예:

```json
"summary": {
  "domain": "economy",
  "total_claims": 1,
  "verifiable_claims": 1,
  "non_verifiable_claims": 0,
  "external_verdicts": {
    "contradicted": 1
  },
  "internal_quality_flags": {
    "multi_number_consistency_check": 1
  }
}
```

---

## 3. `report_payload` 구조

프론트/백엔드에서 쓰기 쉬운 요약 JSON입니다.

형식:

```json
{
  "title": "string",
  "domain": "string",
  "overall_assessment": "string",
  "claim_count": 0,
  "verifiable_claim_count": 0,
  "external_verdicts": {},
  "internal_issue_count": 0,
  "top_external_issues": [],
  "top_internal_issues": []
}
```

### 필드 설명

- `title`
  - 기사 제목
- `domain`
  - 기사 도메인
- `overall_assessment`
  - 전체 요약 한 줄
- `claim_count`
  - 총 claim 수
- `verifiable_claim_count`
  - 검증 가능한 claim 수
- `external_verdicts`
  - verdict 개수 요약
- `internal_issue_count`
  - 내부 품질 경고 수
- `top_external_issues`
  - 사람이 읽기 쉬운 상위 외부 판정 문자열 목록
- `top_internal_issues`
  - 사람이 읽기 쉬운 상위 내부 품질 경고 문자열 목록

예:

```json
"report_payload": {
  "title": "정부, 4월 물가 상승률 2.3% 발표",
  "domain": "economy",
  "overall_assessment": "외부 근거와 충돌하는 claim이 있습니다.",
  "claim_count": 1,
  "verifiable_claim_count": 1,
  "external_verdicts": {
    "contradicted": 1
  },
  "internal_issue_count": 1,
  "top_external_issues": [
    "- [external_fact] claim_contradicted: 주장 수치와 증거 수치가 일치하지 않습니다. (반박됨, 근거: 통계청 보도자료)"
  ],
  "top_internal_issues": [
    "- [internal_quality] multi_number_consistency_check: 하나의 문장에 여러 수치가 있어 추가 대조가 필요합니다. (해당 없음)"
  ]
}
```

---

## 4. `report_payload.llm_report` 구조

LLM 보강이 켜졌을 때만 추가될 수 있습니다.

형식:

```json
{
  "overall_assessment": "string",
  "user_summary": "string",
  "key_points": []
}
```

예:

```json
"llm_report": {
  "overall_assessment": "일부 주장은 외부 근거와 충돌합니다.",
  "user_summary": "이 기사에는 사실과 맞지 않거나 추가 검증이 필요한 부분이 포함되어 있습니다.",
  "key_points": [
    "물가 수치가 공식 발표와 다릅니다.",
    "일부 결론은 근거 대비 과장되어 보입니다."
  ]
}
```

---

## 5. `report_text`

- 타입: `string`
- 의미: 최종 사용자에게 보여줄 문자열

### LLM 미사용 시

`explanation.py`에서 생성한 긴 문자열

예:

```json
"report_text": "제목: 정부, 4월 물가 상승률 2.3% 발표\n분야: economy\n총 claim 수: 1\n검증 가능 claim 수: 1\n총평: 외부 근거와 충돌하는 claim이 있습니다.\n..."
```

### LLM 사용 시

보통 `llm_report.user_summary`가 들어갑니다.

예:

```json
"report_text": "이 기사에는 외부 근거와 충돌하거나 추가 검증이 필요한 부분이 포함되어 있습니다."
```

---

## 6. LLM 내부 structured output 스키마

최종 출력은 아니지만, LLM이 내부적으로 반환하는 JSON 구조는 아래와 같습니다.

### 6-1. Claim Extraction

```json
{
  "claims": [
    {
      "text": "string",
      "claim_type": "string",
      "verifiable": true,
      "rationale": "string"
    }
  ]
}
```

### 6-2. Internal Review

```json
{
  "issues": [
    {
      "label": "string",
      "severity": "string",
      "claim_text": "string",
      "reason": "string"
    }
  ]
}
```

### 6-3. LLM User Report

```json
{
  "overall_assessment": "string",
  "user_summary": "string",
  "key_points": ["string"]
}
```

---

## 7. 축약 예시

```json
{
  "domain": "economy",
  "queries": [
    "정부, 4월 물가 상승률 2.3% 발표"
  ],
  "raw_search_results": [
    {
      "title": "통계청 보도자료",
      "link": "https://example.org/source",
      "snippet": "2026년 4월 소비자물가 상승률은 2.1%였다."
    }
  ],
  "evidence_documents": [
    {
      "title": "통계청 보도자료",
      "text": "2026년 4월 소비자물가 상승률은 2.1%였다.",
      "url": "https://example.org/source",
      "source_type": "trusted_source",
      "domain": "economy",
      "published_at": "2026-05-01"
    }
  ],
  "analysis": {
    "domain": "economy",
    "title": "정부, 4월 물가 상승률 2.3% 발표",
    "claim_count": 1,
    "claims": [
      {
        "text": "정부는 2026년 4월 소비자물가 상승률이 2.3%라고 밝혔다.",
        "sentence_index": 0,
        "domain": "economy",
        "claim_type": "numeric_stat",
        "verifiable": true,
        "rationale": "도메인별 검증 가능한 키워드가 포함되어 있습니다.",
        "numbers": ["2026", "4", "2.3%"],
        "dates": ["2026년 4월"],
        "entities": []
      }
    ],
    "issues": [
      {
        "check_type": "external_fact",
        "label": "claim_contradicted",
        "severity": "high",
        "sentence_index": 0,
        "claim_text": "정부는 2026년 4월 소비자물가 상승률이 2.3%라고 밝혔다.",
        "reason": "주장 수치와 증거 수치가 일치하지 않습니다.",
        "evidence": [
          {
            "title": "통계청 보도자료",
            "url": "https://example.org/source",
            "text": "2026년 4월 소비자물가 상승률은 2.1%였다."
          }
        ],
        "verdict": "contradicted"
      }
    ],
    "summary": {
      "domain": "economy",
      "total_claims": 1,
      "verifiable_claims": 1,
      "non_verifiable_claims": 0,
      "external_verdicts": {
        "contradicted": 1
      },
      "internal_quality_flags": {}
    }
  },
  "report_payload": {
    "title": "정부, 4월 물가 상승률 2.3% 발표",
    "domain": "economy",
    "overall_assessment": "외부 근거와 충돌하는 claim이 있습니다.",
    "claim_count": 1,
    "verifiable_claim_count": 1,
    "external_verdicts": {
      "contradicted": 1
    },
    "internal_issue_count": 0,
    "top_external_issues": [
      "- [external_fact] claim_contradicted: 주장 수치와 증거 수치가 일치하지 않습니다. (반박됨, 근거: 통계청 보도자료)"
    ],
    "top_internal_issues": []
  },
  "report_text": "외부 근거와 충돌하는 claim이 있습니다."
}
```

---

## 8. 한 줄 요약

최종 출력 JSON은 크게 아래 용도로 나뉩니다.

- `queries`, `raw_search_results`, `evidence_documents`
  - 검색/근거 추적용
- `analysis`
  - claim 단위 상세 판정용
- `report_payload`
  - API/프론트 요약용
- `report_text`
  - 사람에게 보여주는 최종 문장
