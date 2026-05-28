# Domain Fact Check MVP

`경제`, `사회`, `과학`, `생활/문화` 4개 분야 전용 뉴스 팩트체크 모듈입니다.

현재 구조는 두 가지 모드로 동작합니다.

- 기본 모드: 규칙 기반 + 검색 기반 파이프라인
- 보강 모드: OpenAI LLM API를 추가한 claim 추출 / 내부 품질 검토 / 설명 생성

현재 포함 기능:

- 분야 라우팅 또는 강제 도메인 지정
- 문장 단위 claim 추출
- 검증 가능성 분류
- 내부 품질 검토
  - 주장 과장 위험
  - 단일 출처 일반화 위험
  - 인과 비약 위험
  - 다중 수치 문장 점검
- 외부 증거 문서와의 비교 인터페이스
  - `supported`
  - `contradicted`
  - `misleading`
  - `unverified`
- 검색 결과 정규화
  - raw search result -> ranked `evidence.json`
- 사용자용 결과 설명 생성
  - analysis json -> report text/json
- OpenAI LLM 보강 옵션
  - structured outputs 기반 claim 추출
  - 내부 품질 검토 보강
  - 사용자 설명 생성 보강
- Solar enhancer 뼈대
  - 동일 인터페이스로 교체 가능
- 실제 search provider 템플릿
  - Tavily
  - SerpAPI

주의:

- 이 모듈은 검색기나 크롤러를 포함하지 않습니다.
- 외부 검증은 별도 검색 단계에서 모은 결과를 정규화해서 `--evidence-json`으로 넘겨주는 구조입니다.
- 제목-본문 불일치는 기존 클릭베이트 기능과 중복되어 제외했습니다.
- API 키는 코드에 직접 넣지 않고 `.env` 파일로 관리하도록 정리했습니다.

## 빠른 이해

이 모듈은 뉴스 기사에 대해:

1. claim 추출
2. 검색 결과를 evidence로 정규화
3. claim/evidence 비교
4. 내부 품질 검토
5. 사용자용 설명 생성

을 수행하는 **분석기**입니다.

이 모듈이 하지 않는 일:

- DB 읽기/쓰기
- 기사 크롤링
- 검색 provider 운영 설정

즉 실서비스에서는 보통:

`DB/수집 시스템 -> domain_fact_check -> DB 저장`

흐름으로 연결합니다.

## RAG와 다른 점

이 모듈은 흔히 말하는 전형적인 RAG와 다릅니다.

- 벡터DB 전제 없음
- 검색 목적이 답변 생성이 아니라 claim 검증
- 결과가 자유 응답이 아니라 verdict/evidence 중심
- 내부 품질 검토 모듈이 따로 있음
- 도메인별 검증 전략을 사용

비교:

- 전형적 RAG:
  - `질문 -> retrieval -> answer generation`
- 이 모듈:
  - `기사 -> claim 추출 -> evidence 검색 -> 비교 -> verdict -> 설명`

## 분야별 검색 신뢰도 규칙

이 모듈은 검색 결과를 아무 문서나 동일하게 취급하지 않습니다.
현재 구현은 **분야별 신뢰도 힌트 + evidence 점수화** 방식입니다.

즉:

- 어떤 claim을 검증 대상으로 볼지 먼저 거르고
- 분야별 신뢰 출처 힌트를 기준으로 검색어를 만들고
- 검색 결과를 점수화해서 더 신뢰도 높은 evidence를 우선 사용합니다.

### 1. 검증 가능한 claim만 우선 처리

분야별로 아래 3가지를 따로 둡니다.

- `verifiable_keywords`
- `weak_keywords`
- `non_verifiable_keywords`

예:

- `economy`
  - 검증 키워드: 금리, 환율, 물가, 실업률, 공시, GDP
  - 약한 표현: 전망, 우려, 예상, 가능성
  - 비검증 표현: 익명, 관계자, 카더라, 소문

- `society`
  - 검증 키워드: 경찰, 소방, 브리핑, 발표, 집계
  - 약한 표현: 추정, 정황, 제보, 목격
  - 비검증 표현: 익명, 관계자, 제보자

- `science`
  - 검증 키워드: 논문, 연구, 실험, 저널, 학회, 데이터
  - 약한 표현: 가능성, 기대, 획기적, 혁신, 추정

- `lifestyle_culture`
  - 검증 키워드: 개봉, 출간, 공연, 전시, 출시, 수상
  - 약한 표현: 인기, 화제, 논란, 감성, 호평

즉 익명 관계자발, 전망성 문장, 해석성 표현이 강한 문장은
검증 우선순위를 낮추거나 `verifiable=false`로 분류합니다.

### 2. 분야별 신뢰 출처 힌트 사용

각 분야마다 신뢰 출처 힌트를 둡니다.

예:

- `economy`
  - 통계청, 한국은행, 금융감독원, DART, 기획재정부
- `society`
  - 경찰청, 소방청, 행정안전부, 질병관리청
- `science`
  - Nature, Science, Cell, PubMed, arXiv, KIST, KAIST
- `lifestyle_culture`
  - 영화진흥위원회, 예스24, 교보문고, 식품의약품안전처, 문화체육관광부

이 힌트는 두 군데에 반영됩니다.

1. 검색어 생성
2. evidence 랭킹

예를 들어 경제 기사면:

- `정부, 4월 물가 상승률 2.3% 발표 통계청`
- `정부, 4월 물가 상승률 2.3% 발표 한국은행`

처럼 신뢰 출처를 붙인 검색어를 추가로 생성합니다.

### 3. 검색 결과 점수화 기준

검색 결과를 evidence로 정규화할 때 아래 기준으로 점수를 매깁니다.

- `trusted_source` 또는 `official_site`면 가산점
- 날짜(`published_at`)가 있으면 가산점
- 분야별 검증 키워드가 많이 맞으면 가산점
- 분야별 신뢰 출처 힌트가 제목/본문/URL에 있으면 가산점
- claim의 숫자/날짜/엔티티가 직접 맞으면 가산점

즉 현재는:

**공식기관/공신력 힌트 + 날짜 + 분야 적합성 + claim 직접 일치도**

를 합쳐서 상위 evidence를 고릅니다.

### 4. 현재 규칙의 성격

중요한 점은, 현재 구현은 **강한 차단형 필터**보다는 **랭킹 중심의 1차 신뢰도 제어**라는 것입니다.

즉:

- 블로그나 일반 뉴스도 입력될 수는 있음
- 다만 공식기관/공신력 높은 출처가 점수상 우선됨

아직 기본 구현에 없는 것:

- 분야별 allowlist 강제
- 분야별 blacklist 강제
- 특정 공식 도메인만 hard filter
- 최소 2개 독립 출처 요구
- paywall/재배포 기사 제거

따라서 현재 README 기준으로는:

- **신뢰도 규칙은 있음**
- **하지만 완전한 출처 통제 시스템은 아직 아님**

## 성능/상태

현재 이 모듈은 **정량 성능 수치가 확정된 학습 분류 모델**이 아니라,
구조 검증과 기능 구현이 완료된 **팩트체크 파이프라인 MVP**입니다.

즉:

- Accuracy/F1 같은 고정 수치는 아직 없음
- rule-based + retrieval + optional LLM 구조
- 경제/과학 분야가 상대적으로 유리
- 사회/생활·문화는 검증 가능한 claim 위주로 해석해야 함

현재 확인된 것은:

- mock 검색 기준 end-to-end 실행 가능
- evidence 정규화 가능
- claim/evidence verdict 생성 가능
- report text / JSON 반환 가능

정량 평가는 이후 별도 평가셋 구축이 필요합니다.

## 엑셀 배치 평가

`news_result.xlsx` 같은 기사 목록 파일에 대해 배치 실행용 평가 스크립트를 포함합니다.

파일:

- `evaluate_news_result.py`
- `evaluate_stored_fact_check_results.py`

현재 이 스크립트가 바로 계산할 수 있는 것:

- 기사 수
- 분야 분포
- article-level verdict 분포
- 평균 claim 수
- 평균 검증 가능 claim 수
- 평균 내부 품질 경고 수
- 내부 품질 경고 분포

중요:

- 현재 엑셀에 `gold_verdict` 같은 정답 라벨이 없으면 **진짜 accuracy/F1은 계산할 수 없습니다.**
- 정답 라벨이 없을 때는 구조/분포/커버리지 지표만 계산합니다.
- 정답 라벨 컬럼이 있으면 article-level accuracy를 같이 계산합니다.

이미 `fact_check_results` 컬럼에 JSON이 저장된 엑셀이라면 아래 스크립트를 씁니다.

- `evaluate_stored_fact_check_results.py`

이 스크립트가 계산하는 proxy 지표 예:

- `fact_check_present_rate`
- `parse_success_rate`
- `json_completeness_rate`
- `domain_distribution`
- `external_verdict_distribution`
- `avg_claim_count`
- `avg_verifiable_claim_ratio`
- `avg_evidence_doc_count`
- `evidence_coverage_rate`
- `trusted_evidence_article_rate`
- `trusted_evidence_doc_rate`

이 지표들은 **구현 신뢰성 / evidence 커버리지 / 운영 품질**을 보여주지만,
**정답 기반 accuracy/F1과는 다릅니다.**

예시 실행:

현재 작업 폴더가 `ai_features/fact_check`일 때

```powershell
py domain_fact_check/evaluate_news_result.py --input ..\..\news_result.xlsx --output-json domain_fact_check/examples/news_result_eval.json
```

카테고리 매핑 JSON이 있으면:

```powershell
py domain_fact_check/evaluate_news_result.py --input ..\..\news_result.xlsx --category-map-json domain_fact_check/examples/category_map.sample.json --output-json domain_fact_check/examples/news_result_eval.json
```

이미 저장된 `fact_check_results` 기준으로 proxy 지표만 집계:

```powershell
py domain_fact_check/evaluate_stored_fact_check_results.py --input ..\..\news_result.xlsx --output-json domain_fact_check/examples/stored_fact_check_eval.json
```

이미 JSON 파일(`news_data.json`)로 받았다면:

```powershell
py domain_fact_check/evaluate_news_data_json.py --input ..\..\news_data.json --output-json domain_fact_check/examples/news_data_eval.json
```

NULL 제외 + `JSON 결과` / `자연어 결과` 분리해서 수동 검토용 파일 생성:

```powershell
py domain_fact_check/prepare_manual_review_from_news_data.py --input ..\..\news_data.json --start-idx 31 --output-prefix domain_fact_check/examples/manual_review
```

이 스크립트는 아래 파일을 만듭니다.

- `manual_review_json_cases.json`
- `manual_review_text_cases.json`
- `manual_review_rerun_candidates.jsonl`

`idx 31` 이후 자연어 요약만 저장된 기사들을 실제로 다시 돌려 4개 verdict를 재생성하려면:

```powershell
py domain_fact_check/rerun_news_data_fact_check.py --input ..\..\news_data.json --start-idx 31 --search-provider tavily --output-json domain_fact_check/examples/rerun_eval_from_31.json --log-path domain_fact_check/examples/rerun_from_31.log
```

## 단계별 로그 확인

교수님 요구처럼 단계별로 LLM과 파이프라인 수행 상태를 확인하려면
`run_fact_check_service()` 또는 `run_fact_check_with_evidence()` 호출 시 `log_path`를 넘기면 됩니다.

로그에 찍히는 단계 예:

- `factcheck_start`
- `domain_inferred`
- `claims_ready`
- `queries_generated`
- `search_query_done`
- `evidence_ranked`
- `analysis_done`
- `llm_extract_claims_start`
- `llm_extract_claims_done`
- `llm_internal_review_start`
- `llm_internal_review_done`
- `llm_user_report_start`
- `llm_user_report_done`
- `factcheck_done`

즉 아래를 확인할 수 있습니다.

- LLM이 실제로 claim 추출을 수행했는지
- 몇 개 claim을 뽑았는지
- 검색 쿼리가 몇 개 생성됐는지
- 검색 결과가 몇 건 왔는지
- evidence가 몇 개 남았는지
- 최종 issue와 summary가 어떻게 나왔는지

예시:

```python
payload = run_fact_check_service(
    title=title,
    body=body,
    category=category,
    search_adapter=search,
    llm_enhancer=llm,
    log_path="domain_fact_check/examples/factcheck_debug.log",
)
```

## 출력 결과 형식

이 모듈의 출력은 단순히 `팩트다 / 아니다` 한 줄로 끝나는 구조가 아닙니다.
기본적으로 **구조화된 JSON**을 반환합니다.

최상위 구조:

```json
{
  "domain": "...",
  "queries": [...],
  "raw_search_results": [...],
  "evidence_documents": [...],
  "analysis": {...},
  "report_payload": {...},
  "report_text": "..."
}
```

### 최상위 필드 의미

- `domain`
  - 기사에 적용된 분야
- `queries`
  - 검색 API에 실제로 사용한 검색어 목록
- `raw_search_results`
  - 검색엔진이 반환한 원본 결과
- `evidence_documents`
  - 검색 결과를 evidence 구조로 정규화한 문서 목록
- `analysis`
  - claim 단위 상세 분석 결과
- `report_payload`
  - 프론트/백엔드에서 쓰기 쉬운 요약 JSON
- `report_text`
  - 사람이 읽기 쉬운 최종 설명 문자열

### 핵심 판정은 어디에 있나

핵심 판정은 `analysis.issues` 안에 들어갑니다.

`issues`는 크게 두 종류입니다.

1. 외부 팩트체크 결과
2. 내부 품질 검토 결과

외부 팩트체크 예시:

```json
{
  "check_type": "external_fact",
  "label": "claim_contradicted",
  "severity": "high",
  "claim_text": "...",
  "reason": "주장 수치와 증거 수치가 일치하지 않습니다.",
  "evidence": [
    {
      "title": "...",
      "url": "...",
      "text": "..."
    }
  ],
  "verdict": "contradicted"
}
```

외부 팩트체크의 `verdict` 값:

- `supported`
- `contradicted`
- `misleading`
- `unverified`

즉 단순 이진 판정이 아니라 **4가지 verdict**를 사용합니다.

내부 품질 검토 예시:

```json
{
  "check_type": "internal_quality",
  "label": "multi_number_consistency_check",
  "severity": "low",
  "claim_text": "...",
  "reason": "하나의 문장에 여러 수치가 있어 추가 대조가 필요합니다.",
  "verdict": "not_applicable"
}
```

내부 품질 검토는 팩트 여부가 아니라 기사 내부 표현 품질에 대한 경고입니다.

예:

- 과장 위험
- 인과 비약 위험
- 단일 출처 일반화 위험
- 다중 수치 점검 필요

### 기사 전체 요약은 어디에 있나

기사 단위 요약은 `analysis.summary`와 `report_payload`에 들어갑니다.

예시:

```json
"summary": {
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

즉:

- claim 총 개수
- 검증 가능한 claim 개수
- 외부 verdict 집계
- 내부 품질 경고 집계

를 기사 단위로 요약합니다.

### 사용자 화면에 바로 쓰는 값

실무적으로 많이 쓰는 값은 아래 2개입니다.

- `report_payload`
  - API 응답 / UI 요약용 JSON
- `report_text`
  - 사용자에게 그대로 보여줄 수 있는 설명 문자열

한 줄 요약:

**이 모듈은 단순 boolean이 아니라, 증거와 판정이 포함된 구조화 JSON을 반환합니다.**

## 폴더 구조

```text
domain_fact_check/
├── ARCHITECTURE_INTEGRATION.md
├── HANDOFF_OVERVIEW.md
├── INTEGRATION_SPEC.md
├── LLM_SETUP_AND_USAGE.txt
├── QUICKSTART_COPYPASTE.txt
├── .env.example
├── analyze_article.py
├── build_evidence.py
├── examples/
├── explain_analysis.py
├── README.md
├── run_service_demo.py
├── TESTING_GUIDE.txt
└── src/
    ├── __init__.py
    ├── evidence_builder.py
    ├── explanation.py
    ├── external_verifier.py
    ├── internal_quality.py
    ├── llm_adapter.py
    ├── llm_enhancer.py
    ├── llm_schemas.py
    ├── models.py
    ├── pipeline.py
    ├── search_adapter.py
    ├── search_providers.py
    ├── service.py
    ├── solar_enhancer.py
    ├── taxonomy.py
    └── utils.py
```

## 지원 도메인

- `economy`
- `society`
- `science`
- `lifestyle_culture`

## 실행

## 담당자가 가장 쉽게 확인하는 방법

처음 받는 사람이 가장 빨리 확인하려면 아래 순서를 따릅니다.

1. `ai_features/fact_check`로 이동
2. 가상환경 생성 및 패키지 설치
3. `domain_fact_check/.env` 생성
4. `domain_fact_check/examples/mock_search_map.sample.json` 기준 mock 테스트 실행
5. 결과 JSON 확인

이 흐름을 짧게 정리한 문서는:

- `HANDOFF_OVERVIEW.md`
- `QUICKSTART_COPYPASTE.txt`

## 설치

이 문서의 모든 명령은 `ai_features/fact_check` 폴더를 현재 작업 폴더로 두는 기준입니다.

VSCode 터미널(Windows PowerShell 기준)에서는 먼저 아래처럼 이동한 뒤 실행합니다.

```powershell
cd ai_features/fact_check
py -m venv .venv
.\.venv\Scripts\Activate.ps1
py -m pip install -r domain_fact_check/requirements.txt
```

## API 키 설정

`.env.example`를 참고해서 `.env` 파일을 만든 뒤 키를 넣으면 됩니다.

권장 위치:

- `domain_fact_check/.env`

예시:

```env
OPENAI_API_KEY=your_openai_api_key
TAVILY_API_KEY=your_tavily_api_key
SERPAPI_API_KEY=your_serpapi_api_key
UPSTAGE_API_KEY=your_upstage_api_key
UPSTAGE_BASE_URL=your_upstage_base_url
```

`.env`, `.env.example` 모두 `.gitignore`에서 제외되도록 설정되어 있습니다.

실험 순서:

1. `.env.example` 파일 열기
2. 같은 폴더에 `.env` 파일 만들기
3. 필요한 키만 채우기
4. 아래 예시 명령으로 테스트 실행

본문을 직접 넣는 경우:

```powershell
py domain_fact_check/analyze_article.py `
  --domain economy \
  --title "정부, 4월 물가 상승률 2.3% 발표" `
  --body "정부는 4월 소비자물가 상승률이 2.3%라고 밝혔다. 전문가들은 이번 수치가 경기 회복을 완전히 입증한다고 말했다." `
  --pretty
```

본문 파일을 쓰는 경우:

```powershell
py domain_fact_check/analyze_article.py `
  --domain science \
  --title "국내 연구진, 신약 후보 효과 입증" `
  --body-file article.txt `
  --pretty
```

외부 증거 문서를 함께 넣는 경우:

```powershell
py domain_fact_check/analyze_article.py `
  --domain economy \
  --title "정부, 4월 물가 상승률 2.3% 발표" `
  --body-file article.txt `
  --evidence-json evidence.json `
  --pretty
```

검색 API raw result를 evidence 문서로 바꾸는 경우:

```powershell
py domain_fact_check/build_evidence.py `
  --domain economy \
  --title "정부, 4월 물가 상승률 2.3% 발표" `
  --body-file article.txt `
  --search-json raw_search_results.json `
  --output evidence_bundle.json `
  --pretty
```

mock search adapter까지 포함해 end-to-end 실행하는 경우:

```powershell
py domain_fact_check/run_service_demo.py `
  --category economy \
  --title "정부, 4월 물가 상승률 2.3% 발표" `
  --body-file domain_fact_check/examples/sample_article_economy.txt `
  --search-json domain_fact_check/examples/mock_search_map.sample.json `
  --pretty
```

OpenAI LLM 보강을 코드에서 사용하는 방식:

```python
from src import OpenAILLMEnhancer, TavilySearchAdapter, run_fact_check_service

llm = OpenAILLMEnhancer(model="gpt-5.4-mini")
search = TavilySearchAdapter()

payload = run_fact_check_service(
    title=title,
    body=body,
    category="economy",
    search_adapter=search,
    llm_enhancer=llm,
)
```

Solar 뼈대를 코드에서 사용하는 방식:

```python
from src import SolarLLMEnhancer

solar = SolarLLMEnhancer(model="solar-mini")
```

주의:

- 현재 `SolarLLMEnhancer`는 인터페이스 뼈대만 구현되어 있습니다.
- 실제 Solar API 호출 로직은 아직 채워야 합니다.

분석 결과를 사용자용 설명으로 바꾸는 경우:

```powershell
py domain_fact_check/explain_analysis.py `
  --analysis-json analysis.json `
  --format text
```

`evidence.json` 예시:

```json
[
  {
    "title": "통계청 보도자료",
    "text": "2026년 4월 소비자물가 상승률은 2.3%였다.",
    "url": "https://example.org/source",
    "source_type": "official_release",
    "domain": "economy",
    "published_at": "2026-05-01"
  }
]
```

`raw_search_results.json` 예시:

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

실제 포함된 샘플 파일:

- `domain_fact_check/examples/sample_article_economy.txt`
- `domain_fact_check/examples/mock_search_map.sample.json`
- `domain_fact_check/examples/raw_search_results.sample.json`

## 출력 해석

- `claims`: 문장별 claim 메타데이터
- `issues`: 내부 품질 경고와 외부 검증 결과
- `summary.external_verdicts`: 외부 근거 기반 판정 개수
- `summary.internal_quality_flags`: 내부 품질 경고 개수
- `build_evidence.py`: 검색 결과를 점수화하고 상위 evidence 후보를 고릅니다.
- `explain_analysis.py`: 분석 결과를 사용자 응답용 텍스트 또는 JSON으로 바꿉니다.

## 다음 단계

실서비스로 연결하려면 아래가 추가로 필요합니다.

1. 사용할 search provider 하나 선택
2. HTML/PDF 본문 추출기
3. 증거 문장 추출기 개선
4. FastAPI/DB 연동

정리:

- OpenAI 설정과 사용법: `LLM_SETUP_AND_USAGE.txt`
- 테스트 절차: `TESTING_GUIDE.txt`
- 백엔드/DB handoff 계약: `INTEGRATION_SPEC.md`
- 아키텍처 연결 위치: `ARCHITECTURE_INTEGRATION.md`
- 빠른 인수인계 요약: `HANDOFF_OVERVIEW.md`
- 복붙용 실행 명령: `QUICKSTART_COPYPASTE.txt`
