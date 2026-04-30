# Clickbait Detection - LightGBM / XGBoost (TF-IDF + SVD)

GBDT 계열(`LightGBM`, `XGBoost`)을 텍스트 벡터화 파이프라인과 결합해
클릭베이트 분류 성능을 비교하기 위한 프로젝트입니다.

## 데이터 및 라벨

- 입력 split: `data/splits/{train,valid,test}.csv`
- 입력 텍스트: `concat_titles = newTitle [SEP] newsTitle`
- 라벨 정의:
  - `label=1`: clickbait
  - `label=0`: nonclickbait

## 데이터 분할 방식

- 방식: 기존 SVM 프로젝트와 동일한 `stratified 80/10/10`
- 샘플 수:
  - Train: 469,392 (clickbait 236,838 / nonclickbait 232,554)
  - Valid: 58,674 (clickbait 29,605 / nonclickbait 29,069)
  - Test: 58,675 (clickbait 29,605 / nonclickbait 29,070)

## 파이프라인

- `word(1,2) TF-IDF + char(2,5) TF-IDF`
- `TruncatedSVD`
- `Normalizer`
- 분류기:
  - `LGBMClassifier` 또는
  - `XGBClassifier`

## 결과 및 진행 상태

현재는 **2단계 방식**으로 진행:
1. Stage-1 빠른 탐색(샘플 데이터)
2. Stage-2 전체 데이터 확정 학습

### Stage-1 빠른 탐색 결과 (완료)
- 최고 조합: `xgb_fast_2`
- 샘플 탐색 기준 점수:
  - Macro F1 `0.7217`
  - Accuracy `0.7218`

### Stage-2 전체 확정 학습 (미완료)
- 로컬 CPU/메모리 환경에서 학습 시간이 과도하게 길어 중단됨
- 따라서 현재 저장된 최종 GBDT 모델 파일은 없음

## 완료/미완료 요약

- 완료:
  - 공통 파이프라인 코드 구성
  - Stage-1 빠른 탐색 실행
- 미완료:
  - Stage-2 전체 데이터 최종 학습 완료
  - 최종 성능(Validation/Test) 확정

## 실행

```bash
py -m pip install -r requirements.txt

# Stage-1 빠른 탐색
py tune.py \
  --train-data data/splits/train.csv \
  --valid-data data/splits/valid.csv \
  --text-col text \
  --label-col label \
  --sample-ratio 0.25 \
  --out models/best_config.json

# Stage-2 전체학습(예시)
py train.py \
  --model-type xgboost \
  --train-data data/splits/train.csv \
  --valid-data data/splits/valid.csv \
  --test-data data/splits/test.csv \
  --svd-components 180 \
  --n-estimators 220 \
  --learning-rate 0.06 \
  --max-depth 9 \
  --model-out models/xgboost_tfidf_svd_best.joblib
```

## 참고

- 현재까지 전체 완료 기준 최고 성능 모델은 `ai_features/clickbait_detection/clickbait_svm_linear`의 튜닝된 `LinearSVC`입니다.
