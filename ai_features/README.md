# AI Features

- clickbait_detection: 낚시성/클릭베이트 탐지 관련 모델
  - current best: `klue/roberta-base`, Test Macro F1 `0.8031`
- political_bias_analysis: 정치 뉴스 편향도 분석 모델
  - current best: `classification/bias_transformer_kopolitic_3class`, Test Macro F1 `0.7276`
- fact_check: 뉴스 팩트체크 관련 모델
  - RAG/검색 증거 연결 실험 폴더

현재 구조:
- `ai_features/clickbait_detection`: 클릭베이트 탐지 모델들
- `ai_features/political_bias_analysis`: 정치 뉴스 편향도 분석 작업 폴더
- `ai_features/fact_check`: 팩트체크 관련 모델과 RAG 실험 폴더
