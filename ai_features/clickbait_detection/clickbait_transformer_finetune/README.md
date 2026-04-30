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
  --model-name models/klue_roberta_clickbait \
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
- Hub repo id를 `--model-name`으로 바로 넣을 수 있습니다.


## Hugging Face Hub usage

`predict_transformer.py`는 로컬 폴더 경로와 Hugging Face repo id를 모두 받을 수 있습니다.

예:

```bash
python predict_transformer.py \
  --model-name JaeRED914/klue-roberta-clickbait-base \
  --title "바뀐 제목" \
  --body "기사 본문"
```

Hugging Face에 업로드할 때는 `model.safetensors`, tokenizer 파일, `config.json`이 함께 있어야 합니다.

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
