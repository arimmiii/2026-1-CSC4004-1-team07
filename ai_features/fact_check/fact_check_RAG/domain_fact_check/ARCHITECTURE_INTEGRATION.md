# Fact Check Architecture Integration

`기술 아키텍처.png` 기준으로 현재 팩트체크 MVP를 어디에 붙이는지 정리한 문서입니다.

## 1. 시스템 구조 기준 연결 위치

이미지의 `기술 아키텍처 — 시스템 구조`에서는 다음 흐름이 보입니다.

- Flutter 앱
- FastAPI 서버
- MySQL DB
- `article_BE.py` 수집/AI 파이프라인 루프

현재 저장소에는 FastAPI/DB 실코드는 없고 AI 모듈 위주만 있으므로, 팩트체크는 `article_BE.py` 또는 그에 해당하는 배치 파이프라인 단계에 연결하는 전제로 설계했습니다.

## 2. 기존 AI 파이프라인 기준 삽입 지점

이미지의 `기술 아키텍처 — AI 파이프라인` 흐름은 대략 아래입니다.

`Google RSS -> DB INSERT -> Selenium 본문 크롤링 -> AI 분류 -> DB UPDATE`

팩트체크는 카테고리 분류 이후, 기사 본문 확보가 끝난 시점에 붙이는 게 맞습니다.

권장 흐름:

`Google RSS`
-> `DB INSERT`
-> `Selenium 본문 크롤링`
-> `카테고리 분류`
-> `클릭베이트/편향도 분석`
-> `도메인 팩트체크`
-> `DB UPDATE`

## 3. 도메인 팩트체크 세부 단계

카테고리 분류 결과가 `경제 / 사회 / 과학 / 생활·문화` 중 하나일 때만 실행합니다.

1. 기사 본문 입력
2. claim 추출
3. 검증 가능 claim 선별
4. 검색 API 호출
5. search result 정규화
6. evidence 문서 재랭킹
7. 외부 근거 비교
8. 내부 품질 검토
9. 최종 설명 생성
10. DB 저장

## 4. 현재 코드와 매핑

- 테스트/실행 기준 작업 폴더:
  - `ai_features/fact_check`
- claim 추출 / 도메인 라우팅:
  - `src/pipeline.py`
- 내부 품질 검토:
  - `src/internal_quality.py`
- search result -> evidence 변환:
  - `src/evidence_builder.py`
  - `build_evidence.py`
- 외부 근거 비교:
  - `src/external_verifier.py`
- 사용자 설명 생성:
  - `src/explanation.py`
  - `explain_analysis.py`

## 5. article_BE.py에 붙일 때 필요한 입출력

### 입력

- `title`
- `content`
- `category`
- 검색 API raw results JSON

### 중간 산출물

- `analysis_json`
- `evidence_json`
- `report_text` 또는 `report_payload`

### DB 저장 권장 필드

현재 이미지의 DB 요약에는 `fact_check`만 보이지만, 실제로는 아래 정도로 쪼개는 편이 낫습니다.

- `fact_check_domain`
- `fact_check_summary`
- `fact_check_report`
- `fact_check_verdict_counts`
- `fact_check_internal_flags`
- `fact_check_evidence_json`

최소 저장만 한다면:

- `fact_check`: 최종 설명 문자열 또는 JSON

## 6. RAG 계획과의 관계

이미지 안의 예전 fact-check RAG 계획은 무시해도 됩니다.

현재 MVP는:

- 벡터DB 없음
- 검색 API raw results 사용 가능
- claim/evidence 비교 중심
- 내부 품질 검토 포함

즉 `RAG 중심 구조`가 아니라 `도메인 라우팅 + evidence normalization + verification` 구조입니다.
