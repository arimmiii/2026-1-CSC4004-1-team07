# Clickbait Detection - Transformer Fine-tuning

단일 Transformer 모델 파인튜닝 실험 프로젝트입니다.

## 데이터 및 라벨

- 입력 split: 공통 데이터 사용
  - `../data/train.csv`
  - `../data/valid.csv`
  - `../data/test.csv`
- 입력 컬럼:
  - `title`: `newTitle` 우선, 없으면 `newsTitle`
  - `body`: `newsContent`
- 라벨 정의:
  - `label=1`: clickbait
  - `label=0`: nonclickbait

현재 폴더 구조 기준 위치:
- 상위 기능 폴더: `ai_features/clickbait_detection`
- Transformer 프로젝트: `ai_features/clickbait_detection/clickbait_transformer_finetune`
- split CSV 위치: `ai_features/clickbait_detection/data`

## 데이터 분할 방식

- 기존 프로젝트와 동일한 `stratified reduced split (200k / 25k / 25k)`
- 샘플 수:
  - Train: 200,000
  - Valid: 25,000
  - Test: 25,000

## 파이프라인

1. `AutoTokenizer`로 토크나이즈 (`max_length=128`)
2. `AutoModelForSequenceClassification(num_labels=2)` 파인튜닝
3. Epoch별 Validation 평가 후 best checkpoint 선택
4. 동일 split Test로 최종 성능 확인

기본 모델:
- `klue/roberta-base`
- 비교 실험: `microsoft/deberta-v3-base`

## KLUE RoBERTa base experiment tracking

`KLUE RoBERTa base`는 앞으로 하이퍼파라미터 실험을 다시 돌릴 예정이므로, 기준 설정과 후속 실험 설정을 아래처럼 고정해서 구분합니다.

### Baseline setting

- 목적: 현재 README와 시각화에 반영된 공식 기준 실험
- 모델: `klue/roberta-base`
- 데이터: reduced split `200k / 25k / 25k`
- 하이퍼파라미터:
  - `learning_rate=2e-5`
  - `epochs=2`
  - `max_length=128`
  - `batch_size=8`
  - `warmup_ratio=0.06`
- 권장 저장 이름:
  - `models/klue_roberta_clickbait_title_body`
  - 또는 별도 관리 시 `models/klue_roberta_base_baseline`

### Follow-up runs

| Run ID | learning_rate | epochs | max_length | Purpose |
| --- | ---: | ---: | ---: | --- |
| `A` | `1e-5` | `2` | `128` | lower LR stability check |
| `B` | `2e-5` | `3` | `128` | longer training check |

권장 저장 이름:

- Run A: `models/klue_roberta_base_runA_lr1e5_ep2_len128`
- Run B: `models/klue_roberta_base_runB_lr2e5_ep3_len128`

권장 출력 폴더:

- Run A: `outputs/klue_roberta_base_runA_lr1e5_ep2_len128`
- Run B: `outputs/klue_roberta_base_runB_lr2e5_ep3_len128`

실험 결과를 추가할 때는 각 run 폴더 안의 `metrics.json`을 기준으로 비교하고, baseline과 탐색 run을 섞어서 해석하지 않습니다.

빠른 실험용 설정:
- `max_length=96`
- `epochs=1`
- subset 사용
  - Train: `120,000`
  - Valid: `20,000`
  - Test: `20,000`

## 실행 (Colab 권장)

```bash
pip install -r requirements-colab.txt

python train_transformer.py \
  --model-name klue/roberta-base \
  --train-data ../data/train.csv \
  --valid-data ../data/valid.csv \
  --test-data ../data/test.csv \
  --title-col title \
  --body-col body \
  --max-length 128 \
  --epochs 2 \
  --batch-size 8 \
  --learning-rate 2e-5 \
  --output-dir outputs/klue_roberta_clickbait \
  --save-model-dir models/klue_roberta_clickbait
```

## 예측

```bash
python predict_transformer.py \
  --model-dir models/klue_roberta_clickbait \
  --title "바뀐 제목" \
  --body "기사 본문"
```

## 빠른 실험 결과

- 아래 수치는 subset fast 실험 기록입니다.
- 모델: `klue/roberta-base`
- 실행 노트북: `colab_klue_roberta_finetune_fast.ipynb`
- 체크포인트: `outputs/klue_roberta_clickbait_fast/checkpoint-15000`
- 평가 종류: `subset validation`
- 결과:
  - Macro F1: `0.8241`
  - Weighted F1: `0.8243`
  - Accuracy: `0.8243`

주의:
- 이 결과는 `전체 train/valid/test` 최종 점수가 아니라, 빠른 확인용 `subset + 1 epoch` 실험 결과입니다.
- 최종 제출용 점수는 전체 데이터 학습 후 `test` 기준으로 다시 측정해야 합니다.

## 전체 test 결과

아래 값들은 같은 reduced split(200k / 25k / 25k)에서 다시 학습/평가한 최종 기록입니다.

### KLUE RoBERTa base

- 모델: `klue/roberta-base`
- 저장 위치: `models/klue_roberta_clickbait_title_body`
- 실험 구분: `baseline (lr=2e-5, epochs=2, max_length=128)`
- Validation:
  - Macro F1: `0.8007271615`
  - Weighted F1: `0.8006464703`
  - Accuracy: `0.80112`
- Test:
  - Macro F1: `0.8027548690`
  - Weighted F1: `0.8026818347`
  - Accuracy: `0.80308`

### KLUE RoBERTa base - Run A

- 모델: `klue/roberta-base`
- 저장 위치: `models/klue_roberta_base_runA_lr1e5_ep2_len128`
- 실험 구분: `Run A (lr=1e-5, epochs=2, max_length=128)`
- Validation:
  - Macro F1: `0.8125922389`
  - Weighted F1: `0.8125552527`
  - Accuracy: `0.81268`
- Test:
  - Macro F1: `0.8144898932`
  - Weighted F1: `0.8144570036`
  - Accuracy: `0.81456`

### KLUE RoBERTa base - Run B

- 모델: `klue/roberta-base`
- 저장 위치: `models/klue_roberta_base_runB_lr2e5_ep3_len128`
- 실험 구분: `Run B (lr=2e-5, epochs=3, max_length=128)`
- Validation:
  - Macro F1: `0.8050706492`
  - Weighted F1: `0.8050248543`
  - Accuracy: `0.8052`
- Test:
  - Macro F1: `0.8054167071`
  - Weighted F1: `0.8053758203`
  - Accuracy: `0.80552`

### KLUE RoBERTa base run comparison

| Run | learning_rate | epochs | max_length | Test Macro F1 | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| Baseline | `2e-5` | `2` | `128` | `0.8028` | original final rerun |
| Run A | `1e-5` | `2` | `128` | `0.8145` | current best |
| Run B | `2e-5` | `3` | `128` | `0.8054` | slightly above baseline, below Run A |

### DeBERTa base

- 모델: `microsoft/deberta-v3-base`
- 저장 위치: `models/deberta_v3_base_title_body_run1`
- Validation:
  - Macro F1: `0.7762925295`
  - Weighted F1: `0.7761489802`
  - Accuracy: `0.7774`
- Test:
  - Macro F1: `0.7792306905`
  - Weighted F1: `0.7791000601`
  - Accuracy: `0.78016`

### KLUE RoBERTa large

- 모델: `klue/roberta-large`
- 저장 위치: `models/klue_roberta_large_title_body_run2`
- Validation:
  - Macro F1: `0.3353538576`
  - Weighted F1: `0.3384122848`
  - Accuracy: `0.50456`
- Test:
  - Macro F1: `0.3353538576`
  - Weighted F1: `0.3384122848`
  - Accuracy: `0.50456`

메모:
- `roberta-large`는 다시 학습했지만 최종 채택 기준에서는 제외했습니다.
- `DeBERTa base`는 비교용 transformer 실험으로는 의미가 있었지만, 최종 성능은 `KLUE RoBERTa base`보다 낮았습니다.
- 현재 저장소의 메인 Transformer 결과는 `KLUE RoBERTa base`입니다.

## 완료/미완료 상태

- 완료:
  - 프로젝트 구조/학습 스크립트/추론 스크립트 구성
  - 동일 split 재사용 설정
  - 빠른 실험용 Colab 노트북 구성
  - 빠른 실험 1회 수행 및 validation 점수 확인
- 미완료:
  - 전체 데이터 기준 최종 파인튜닝 실행
  - 전체 `test` 기준 최종 점수 측정
  - SVM/LR과 최종 성능 비교표 확정
