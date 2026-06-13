# AI Features

- clickbait_detection: 낚시성/클릭베이트 탐지 관련 모델
  - rerun snapshot: Linear SVM `0.6773`, LogReg `0.6814`, DeBERTa base `0.7792`, KLUE RoBERTa base baseline `0.8031`, Run A `0.8145`, Run B `0.8054`
  - current best: `klue/roberta-base` Run A, Test Macro F1 `0.8145`
- political_bias_analysis: 정치 뉴스 편향도 분석 모델
  - rerun snapshot: SVM `0.6740`, LogReg `0.6520`, 3-class transformer `0.7276`
  - current best: `classification/bias_transformer_kopolitic_3class`, Test Macro F1 `0.7276`
- fact_check: 뉴스 팩트체크 관련 모델
  - RAG/검색 증거 연결 실험 폴더
  - `domain_fact_check`: 경제/사회/과학/생활·문화용 규칙 기반 MVP 분석기
    - 검색 결과 정규화, 내부 품질 검토, 설명 생성 포함
    - OpenAI LLM 보강 옵션과 Solar enhancer 뼈대 포함

현재 구조:
- `ai_features/clickbait_detection`: 클릭베이트 탐지 모델들
- `ai_features/political_bias_analysis`: 정치 뉴스 편향도 분석 작업 폴더
- `ai_features/fact_check`: 팩트체크 관련 모델과 RAG 실험 폴더
  - `ai_features/fact_check/domain_fact_check`: 4개 분야 전용 claim/근거 분석 MVP
