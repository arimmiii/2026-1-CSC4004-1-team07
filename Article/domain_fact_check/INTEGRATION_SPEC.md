# Domain Fact Check Integration Spec

이 문서는 DB 담당자 또는 백엔드 담당자에게 넘길 때 필요한 입력/출력 계약을 정리합니다.

## 1. 이 기능이 LLM API를 반드시 쓰는가

현재 저장소의 `domain_fact_check`는 **LLM API 없이도 동작**하지만, 이제 **OpenAI LLM API를 선택적으로 붙일 수 있게** 되어 있습니다.

포함된 로직:

- 도메인 라우팅
- claim 추출
- 검증 가능성 판정
- 내부 품질 검토
- 검색 결과 정규화
- evidence 비교
- 사용자용 설명 생성

기본 모드:

- 규칙 기반 + 검색 기반 파이프라인

LLM 보강 모드:

- OpenAI `Responses API`
- structured outputs 기반 claim 추출 / 내부 품질 검토 / 사용자 설명 생성

OpenAI 공식 문서 기준:

- Responses API는 텍스트/이미지 입력과 텍스트/JSON 출력을 지원합니다.
- Python SDK에서는 `client.responses.create(...)`와 `client.responses.parse(...)` 패턴을 사용합니다.
- 최신 모델 선택 가이드는 `gpt-5.5`를 기본 출발점으로 제시하고, 비용/지연이 중요하면 더 작은 GPT-5.4 계열을 권장합니다.

현재 코드 기본값은 비용과 성능 균형을 위해 `gpt-5.4-mini`로 잡았습니다.

API 키는 `.env` 파일에서 자동으로 읽도록 정리했습니다.

권장 위치:

- `domain_fact_check/.env`

LLM API를 붙이면 좋은 지점:

- claim 추출 품질을 더 높이고 싶을 때
- 내부 품질 검토를 더 정교하게 하고 싶을 때
- 최종 설명 문장을 더 자연스럽게 만들고 싶을 때
- 검색 쿼리 rewrite 성능을 올리고 싶을 때

정리:

- **규칙 기반만으로도 동작**
- **OpenAI LLM API를 붙이면 품질을 더 끌어올릴 수 있음**
- **현재 코드에 이미 반영됨**

## 2. 기사 본문은 어디서 오나

네가 말한 방식이 맞습니다.

흐름은 보통:

1. 크롤러/수집기에서 기사 제목과 본문을 DB에 저장
2. 백엔드 또는 배치 파이프라인이 DB에서 `title`, `content`, `category`를 읽음
3. 이 모듈에 입력으로 전달
4. 검색 API 호출 결과와 함께 분석
5. 분석 결과를 다시 DB에 저장

즉 이 기능은 **DB에 저장된 기사 본문을 읽어 사용하는 후처리 AI 모듈**로 보는 게 맞습니다.

## 3. 입력 계약

최소 입력:

```json
{
  "title": "정부, 4월 물가 상승률 2.3% 발표",
  "body": "정부는 2026년 4월 소비자물가 상승률이 2.3%라고 밝혔다.",
  "category": "economy"
}
```

허용 카테고리:

- `economy`
- `society`
- `science`
- `lifestyle_culture`

카테고리를 안 주면 본문 기준으로 내부 추정합니다.

## 4. Search Adapter 계약

백엔드가 실제 provider를 붙일 때 맞춰야 할 최소 인터페이스는 아래입니다.

Python 개념상:

```python
class SearchAdapter(Protocol):
    def search(self, request: SearchRequest) -> list[dict]:
        ...
```

입력:

```json
{
  "query": "정부 4월 소비자물가 상승률 2.3% 통계청",
  "domain": "economy",
  "top_k": 10
}
```

raw result 표준 스키마:

```json
[
  {
    "title": "통계청 보도자료",
    "link": "https://example.org/source",
    "snippet": "2026년 4월 소비자물가 상승률은 2.3%였다.",
    "source": "통계청",
    "published_at": "2026-05-01"
  }
]
```

허용 필드 alias:

- title 계열: `title`, `name`, `headline`
- url 계열: `url`, `link`
- text 계열: `text`, `content`, `body`, `snippet`, `description`
- source 계열: `source_type`, `source`, `site_name`, `publisher`
- date 계열: `published_at`, `date`, `published`, `pubDate`

현재 제공되는 provider 템플릿:

- `TavilySearchAdapter`
- `SerpApiSearchAdapter`

둘 다 실제 네트워크 호출은 백엔드 실행 환경에서 API 키를 넣어 사용하면 됩니다.

`.env` 예시:

```env
TAVILY_API_KEY=your_tavily_api_key
SERPAPI_API_KEY=your_serpapi_api_key
```

## 5. 출력 계약

서비스 wrapper `run_fact_check_service()`의 출력은 아래 구조입니다.

```json
{
  "domain": "economy",
  "queries": ["..."],
  "raw_search_results": [{ "...": "..." }],
  "evidence_documents": [{ "...": "..." }],
  "analysis": { "...": "..." },
  "report_payload": { "...": "..." },
  "report_text": "..."
}
```

핵심 필드 의미:

- `queries`: 검색 API에 사용한 질의들
- `raw_search_results`: provider가 반환한 원본 결과
- `evidence_documents`: 정규화 및 재랭킹된 근거 후보
- `analysis`: claim/issue 단위 상세 결과
- `report_payload`: 프론트/백엔드가 쓰기 쉬운 요약 JSON
- `report_text`: 사용자에게 바로 보여줄 수 있는 설명 문자열

## 6. DB 저장 권장값

DB 담당자에게는 최소 아래를 저장하라고 넘기면 충분합니다.

- `fact_check_domain`
- `fact_check_report`
- `fact_check_summary_json`
- `fact_check_evidence_json`

기존 스키마가 `fact_check` 단일 필드만 허용한다면:

- `fact_check`: `report_text`

또는

- `fact_check`: `report_payload` JSON string

## 7. 현재 코드 진입점

- 로컬 검색 mock 포함 end-to-end 실행:
  - `run_service_demo.py`
- 서비스 wrapper:
  - `src/service.py`
- 검색 결과 정규화만 실행:
  - `build_evidence.py`
- 기사 분석만 실행:
  - `analyze_article.py`
- 사용자 설명 생성만 실행:
  - `explain_analysis.py`
- OpenAI LLM 보강:
  - `src/llm_enhancer.py`
- Solar 뼈대:
  - `src/solar_enhancer.py`
- 실제 search provider 템플릿:
  - `src/search_providers.py`
- `.env` 자동 로딩:
  - `src/config.py`

## 8. 백엔드가 실제로 해야 하는 일

네 기능 밖에서 필요한 건 아래 둘입니다.

1. DB에서 기사 본문 읽기
2. Search API provider 구현

그 외 팩트체크 분석 로직은 현재 폴더에 들어 있습니다.
