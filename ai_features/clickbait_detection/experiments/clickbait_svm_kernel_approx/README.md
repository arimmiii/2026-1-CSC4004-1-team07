# Clickbait Detection - Kernel Approx SVM

`RBFSampler + LinearSVC`로 비선형성을 근사한 SVM 실험 프로젝트입니다.

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

- `word(1,2) TF-IDF`
- `TruncatedSVD`
- `Normalizer`
- `RBFSampler`
- `LinearSVC`

## 결과 (동일 split 기준)

- Validation: Macro F1 `0.5839`, Accuracy `0.5845`
- Test: Macro F1 `0.5838`, Accuracy `0.5843`

모델 파일:
- `models/kernel_approx_svm.joblib`

## 완료/미완료 상태

- 완료:
  - 기본 커널 근사 SVM 학습/평가
- 미완료:
  - 안정적 성능 개선 튜닝(고차원 설정은 시간/메모리 부담 큼)

## 실행

```bash
py -m pip install -r requirements.txt

py train.py \
  --train-data data/splits/train.csv \
  --valid-data data/splits/valid.csv \
  --test-data data/splits/test.csv \
  --text-col text \
  --label-col label \
  --svd-components 256 \
  --rbf-components 512 \
  --gamma 0.7 \
  --c 1.0 \
  --model-out models/kernel_approx_svm.joblib
```

## 예측

```bash
py predict.py \
  --model models/kernel_approx_svm.joblib \
  --text "이 기사 안 보면 평생 후회합니다 [SEP] 정부, 내년도 예산안 발표"
```
