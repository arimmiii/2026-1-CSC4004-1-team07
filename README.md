# AI Features Project

`뉴스 AI 기능`을 중심으로 한 통합 프로젝트입니다. 현재 저장소는 다음 세 가지 축으로 구성됩니다.

- `clickbait_detection`: 낚시성/클릭베이트 기사 탐지
- `political_bias_analysis`: 정치 뉴스 편향도 분석
- `fact_check`: 뉴스 팩트체크 및 RAG 실험

이 저장소는 실제 서비스 연동을 염두에 두고, 각 기능별로 모델/데이터/실험 코드를 분리해 두었습니다.  
세부 실행 방법은 각 하위 폴더의 `README.md`를 참고하면 됩니다.

## Repository Structure

```text
sw_project/
├── ai_features/
│   ├── clickbait_detection/
│   ├── political_bias_analysis/
│   └── fact_check/
└── README.md
```

## 1. Clickbait Detection

클릭베이트 탐지는 현재 가장 완성도가 높은 메인 기능입니다.

### 목표

- 뉴스 제목과 본문을 보고 낚시성 기사인지 분류
- 서비스에서는 기사 수집 후 1차 필터링 또는 검수 보조 용도로 사용

### 데이터

- 입력 컬럼
  - `title`
  - `body`
  - `label`
- 라벨 정의
  - `label=1`: clickbait
  - `label=0`: nonclickbait
- 최종 데이터 분할
  - Train: `200,000`
  - Valid: `25,000`
  - Test: `25,000`
- 분할 방식
  - stratified split
  - 클래스 비율 유지

### 모델

- `clickbait_svm_linear`
  - `word TF-IDF + char TF-IDF + LinearSVC`
- `clickbait_logreg_tfidf`
  - `word TF-IDF + char TF-IDF + LogisticRegression`
- `clickbait_transformer_finetune`
  - `klue/roberta-base`
  - `klue/roberta-large` 실험 노트북
  - `microsoft/deberta-v3-base` 실험 노트북
- `experiments`
  - 성능이 낮았거나 운영 후보에서 제외된 실험

### 현재 기록된 성능

아래 값은 저장소에 남아 있는 최신 결과 기준입니다.  
데이터 구성이나 split이 다르면 수치를 직접 비교하면 안 됩니다.

| Model | Data config | Valid Macro F1 | Test Macro F1 | Accuracy | Status |
| --- | --- | ---: | ---: | ---: | --- |
| Logistic Regression | reduced split 200k/25k/25k | 0.6790 | 0.6814 | 0.6814 | reference baseline |
| KLUE RoBERTa base | reduced split 200k/25k/25k | 0.8011 | 0.8031 | 0.8031 | current clickbait main model |
| KLUE RoBERTa large | reduced split 200k/25k/25k | 0.5046 | 0.5046 | 0.5046 | failed/stable not reached |

주의:

- `Linear SVM` 결과는 현재 저장소의 아카이브 기준선으로만 남아 있습니다.
- `Logistic Regression`과 `KLUE RoBERTa base`는 현재 reduced split 기준 결과입니다.
- `KLUE RoBERTa large`는 현재 설정에서는 수렴에 실패했습니다.

### 추천 사용 방향

- 서비스 비용을 최소화하려면 `Linear SVM`
- 품질을 조금 더 챙기려면 `KLUE RoBERTa base`
- 운영 서버 자원이 넉넉하면 Transformer 계열을 고려

### 주요 파일

- `ai_features/clickbait_detection/clickbait_svm_linear/`
- `ai_features/clickbait_detection/clickbait_logreg_tfidf/`
- `ai_features/clickbait_detection/clickbait_transformer_finetune/`
- `ai_features/clickbait_detection/make_reduced_splits.py`

## 2. Political Bias Analysis

정치 뉴스 편향도 분석 기능입니다.

현재는 다음과 같은 구성으로 분리되어 있습니다.

- `classification`
  - 편향 여부 또는 편향 카테고리 분류
- `regression`
  - 편향 점수 회귀

세부 구현은 하위 README를 따릅니다.

### 현재 기록된 성능 요약

Test 기준 대표값:

| Model | Type | Accuracy | Macro F1 | Top2 |
| --- | --- | ---: | ---: | ---: |
| `classification/bias_transformer_kopolitic_3class` | 3-class classification | 0.7420 | 0.7276 | 0.9280 |
| `classification/bias_svm_linear` | 3-class classification | 0.6740 | 0.6627 | 0.9320 |
| `classification/bias_logreg_tfidf` | 3-class classification | 0.6520 | 0.6387 | 0.9200 |
| `regression/bias_svr_linear` | regression, rounded | 0.3260 | 0.2621 | - |

### 주요 파일

- `ai_features/political_bias_analysis/README.md`
- `ai_features/political_bias_analysis/classification/README.md`
- `ai_features/political_bias_analysis/regression/README.md`

## 3. Fact Check

뉴스 팩트체크 기능과 RAG 실험 폴더입니다.

- `fact_check_RAG`
  - 크롤링
  - 임베딩
  - 검색/증거 연결 실험

### 주요 파일

- `ai_features/fact_check/fact_check_RAG/`

## Reproducibility

### 공통 설치

각 하위 프로젝트 폴더에서 별도 요구사항 파일을 설치합니다.

예:

```bash
python -m pip install -r requirements.txt
```

Transformer는 Colab 기준으로 다음 파일을 사용합니다.

```bash
pip install -r requirements-colab.txt
```

### Clickbait Detection 재현 순서

1. `ai_features/clickbait_detection/data/train.csv`, `valid.csv`, `test.csv` 준비
2. 원하는 하위 모델 폴더로 이동
3. 학습 실행
4. 필요하면 `predict.py`로 단건 예측

예:

```bash
cd ai_features/clickbait_detection/clickbait_svm_linear
python train.py \
  --data data/train.csv \
  --valid-data data/valid.csv \
  --test-data data/test.csv \
  --title-col title \
  --body-col body \
  --label-col label \
  --model-out models/linear_svm_clickbait_title_body.joblib
```

Transformer는 Colab 노트북을 권장합니다.

## Data Policy

이 저장소에는 원본 대용량 데이터와 모델 산출물을 포함하지 않습니다.

제외 대상:

- `data/`
- `models/`
- `outputs/`
- `metrics.json`
- `*.joblib`
- `*.safetensors`
- `trainer_state.json`

## Project Status

- `clickbait_detection`
  - 가장 우선순위가 높은 기능
  - 현재 서비스 연결 가능 수준의 실험 결과가 있음
  - SVM은 아카이브 기준선, 현재 비교 중심은 LogReg / Transformer
- `political_bias_analysis`
  - 기능 분리 및 베이스라인 준비 단계
- `fact_check`
  - RAG 기반 실험 폴더 정리 단계

## Notes

- 하위 폴더 README가 더 상세합니다.
- 실험 수치가 서로 다른 데이터 설정에서 나온 경우, 직접적인 우열 비교는 피해야 합니다.
- 마감 우선이면 추가 탐색보다 기능 연동과 안정성이 우선입니다.
