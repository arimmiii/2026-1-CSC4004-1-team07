# Domain Fact Check Handoff Overview

이 문서는 `domain_fact_check` 모듈을 처음 받는 담당자가 빠르게 이해하고 실행할 수 있도록 정리한 요약 문서입니다.

## 1. 이 모듈이 하는 일

이 모듈은 뉴스 기사에 대해 아래를 수행합니다.

1. 기사 분야 분류 또는 강제 지정
2. 기사 본문에서 claim 추출
3. 검증 가능한 claim 선별
4. 검색 결과를 evidence 후보로 정규화
5. evidence와 claim 비교
6. 내부 품질 검토
7. 사용자용 설명 생성

대상 분야는 아래 4개입니다.

- 경제
- 사회
- 과학
- 생활/문화

## 2. 이 모듈이 하지 않는 일

- DB에서 기사 읽기
- DB에 결과 저장
- 실제 크롤링
- 실제 search provider 운영 설정

즉 이 모듈은 저장/수집 시스템이 아니라 **팩트체크 분석기**입니다.

## 3. 실행 위치

모든 문서는 아래 폴더를 현재 작업 폴더로 두는 기준입니다.

```powershell
cd ai_features/fact_check
```

## 4. 가장 쉬운 확인 방법

### 1단계. 패키지 설치

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
py -m pip install -r domain_fact_check/requirements.txt
```

### 2단계. `.env` 파일 만들기

위치:

- `domain_fact_check/.env`

최소 예시:

```env
OPENAI_API_KEY=YOUR_KEY
```

실제 검색 provider까지 쓸 경우:

```env
OPENAI_API_KEY=YOUR_KEY
TAVILY_API_KEY=YOUR_KEY
SERPAPI_API_KEY=YOUR_KEY
```

### 3단계. mock 검색으로 테스트

```powershell
py domain_fact_check/run_service_demo.py --category economy --title "정부, 4월 물가 상승률 2.3% 발표" --body-file domain_fact_check/examples/sample_article_economy.txt --search-json domain_fact_check/examples/mock_search_map.sample.json --pretty
```

샘플 파일:

- 기사 본문:
  - `domain_fact_check/examples/sample_article_economy.txt`
- mock 검색 결과:
  - `domain_fact_check/examples/mock_search_map.sample.json`

복붙용 전체 명령 모음:

- `QUICKSTART_COPYPASTE.txt`

## 5. 결과를 어디서 확인하나

현재 모듈은 기본적으로 **JSON을 표준 출력(stdout)으로 반환**합니다.

즉:

- 자동 DB 저장 없음
- 자동 파일 저장 없음
- 실행 결과를 백엔드가 받아 저장하는 구조

필요하면 터미널에서 파일로 저장할 수 있습니다.

```powershell
py domain_fact_check/run_service_demo.py --category economy --title "정부, 4월 물가 상승률 2.3% 발표" --body-file domain_fact_check/examples/sample_article_economy.txt --search-json domain_fact_check/examples/mock_search_map.sample.json --pretty > domain_fact_check/examples/output.sample.json
```

## 6. 출력 구조

핵심 출력 필드:

- `queries`
- `raw_search_results`
- `evidence_documents`
- `analysis`
- `report_payload`
- `report_text`

실무에서 많이 보는 필드:

- 사용자 표시용:
  - `report_text`
- API/프론트 전달용:
  - `report_payload`
- 상세 검토/로그 저장용:
  - `analysis`

## 7. 기능 요약

### 외부 팩트체크

- claim과 evidence를 비교
- verdict:
  - `supported`
  - `contradicted`
  - `misleading`
  - `unverified`

### 내부 품질 검토

- 다중 수치 문장 점검
- 과장 가능성
- 단일 출처 일반화 위험
- 인과 비약 위험

### LLM 보강

OpenAI LLM API를 붙이면 아래 품질을 높일 수 있습니다.

- claim 추출
- 내부 품질 검토
- 사용자용 요약 설명

## 8. RAG와 다른 점

이 모듈은 전형적인 RAG 구조가 아닙니다.

RAG와의 차이:

1. 벡터DB를 전제로 하지 않음
2. retrieval의 목적이 답변 생성이 아니라 claim 검증임
3. 검색 결과를 evidence로 정규화하고 판정 라벨을 반환함
4. 내부 품질 검토 모듈이 따로 있음
5. 분야별 검증 전략을 전제로 함

즉 이 모듈은:

- `질문 -> 관련 문서 검색 -> 답변 생성`

이 아니라

- `기사 -> claim 추출 -> evidence 검색 -> claim/evidence 비교 -> 판정 -> 설명`

구조입니다.

## 9. 성능에 대해

현재 저장소 기준 이 모듈은 **학습된 정량 성능 지표가 있는 분류 모델이 아니라 분석 파이프라인 MVP**입니다.

즉:

- Accuracy/F1 같은 확정 수치가 아직 없음
- 현재는 rule-based + retrieval + optional LLM 구조 검증 단계
- 품질은 도메인/기사 유형/검색 결과 품질에 따라 달라짐

따라서 지금 단계에서 말할 수 있는 것은:

- 경제/과학은 비교적 잘 맞는 편
- 사회/생활·문화는 검증 가능 claim에 한정할 때 의미 있음
- mock 기준으로는 전체 파이프라인이 정상 동작

정량 성능을 만들려면 이후 아래가 필요합니다.

1. 도메인별 평가셋 구축
2. claim-level 정답 라벨 구축
3. verdict 기준 평가
4. 내부 품질 경고 precision 점검

## 10. 담당자가 해야 할 일

백엔드/DB 담당자가 실제로 해야 하는 일은 주로 아래입니다.

1. DB에서 기사 본문 읽기
2. search provider 선택 및 API 키 연결
3. `run_fact_check_service()` 호출
4. 반환 JSON을 DB에 저장

## 11. 참고 문서

- 복붙용 빠른 시작:
  - `QUICKSTART_COPYPASTE.txt`
- 전체 사용법: `README.md`
- LLM 설정: `LLM_SETUP_AND_USAGE.txt`
- 테스트 가이드: `TESTING_GUIDE.txt`
- 입출력 계약: `INTEGRATION_SPEC.md`
- 아키텍처 연결: `ARCHITECTURE_INTEGRATION.md`
