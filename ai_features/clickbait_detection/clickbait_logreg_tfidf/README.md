# Clickbait Detection - TF-IDF + Logistic Regression

SVM 대안으로 `TF-IDF + Logistic Regression` 이진 분류를 수행합니다.

## 데이터 및 라벨

- 입력 split: `../data/{train,valid,test}.csv`
- 입력 컬럼:
  - `title`: `newTitle` 우선, 없으면 `newsTitle`
  - `body`: `newsContent`
- 라벨 정의:
  - `label=1`: clickbait
  - `label=0`: nonclickbait

## 데이터 분할 방식

- 방식: 기존 SVM 프로젝트와 동일한 `stratified reduced split (200k / 25k / 25k)`
- 샘플 수:
  - Train: 200,000 (clickbait 100,913 / nonclickbait 99,087)
  - Valid: 25,000 (clickbait 12,614 / nonclickbait 12,386)
  - Test: 25,000 (clickbait 12,614 / nonclickbait 12,386)

## 파이프라인

- 특징 추출: `word(1,2) TF-IDF + char(2,5) TF-IDF`
- 특징 수 제한: `word max_features=50000`, `char max_features=50000`
- 분류기: `LogisticRegression(saga, C=10.0)`

## 결과 (공통 `title/body` 데이터 기준)
- Validation: Macro F1 `0.8145`, Weighted F1 `0.8146`, Accuracy `0.8150`
- Test: Macro F1 `0.8145`, Weighted F1 `0.8146`, Accuracy `0.8150`

모델 파일:
- `models/logreg_tfidf.joblib`

## 완료/미완료 상태

- 완료:
  - C 튜닝(4.0 → 6.0 → 8.0 → 10.0)
  - 최고 성능 모델 저장
- 미완료:
  - 클래스별 임계값 튜닝
  - 운영 환경 지연시간/메모리 벤치마크

## 실행

```bash
py -m pip install -r requirements.txt

py train.py \
  --train-data ../data/train.csv \
  --valid-data ../data/valid.csv \
  --test-data ../data/test.csv \
  --title-col title \
  --body-col body \
  --label-col label \
  --c 10.0 \
  --model-out models/logreg_tfidf.joblib
```

## 예측

```bash
py predict.py \
  --model models/logreg_tfidf.joblib \
  --title "바뀐 제목" \
  --body "기사 본문"
```
