# Model Release Plan

이 문서는 저장소의 학습 결과를 GitHub Release로 분리할 때 쓰는 파일 구성 기준입니다.

## Clickbait Detection

권장 release 이름:

- `clickbait-models-v1`

권장 asset:

- `clickbait_svm_linear.zip`
- `clickbait_logreg_tfidf.zip`
- `klue_roberta_clickbait_title_body.zip`
- `klue_roberta_large_title_body_run2.zip`

각 zip에 넣는 파일:

- SVM / LogReg:
  - 모델 파일 1개
  - `metrics.json`
- Transformer:
  - `config.json`
  - `model.safetensors`
  - `tokenizer.json`
  - `tokenizer_config.json`
  - `special_tokens_map.json` 또는 `vocab.txt`
  - `training_args.bin`
  - `metrics.json`

## Political Bias Analysis

권장 release 이름:

- `bias-models-v1`

권장 asset:

- `bias_svm_linear.zip`
- `bias_logreg_tfidf.zip`
- `bias_ridge_tfidf.zip`
- `bias_svr_linear.zip`
- `bias_kopolitic_3class.zip`
- `bias_transformer_regression.zip`

각 zip에 넣는 파일:

- SVM / LogReg / Ridge / SVR:
  - 모델 파일 1개
  - `metrics.json`
- Transformer:
  - `config.json`
  - `model.safetensors`
  - `tokenizer.json`
  - `tokenizer_config.json`
  - `special_tokens_map.json` 또는 `vocab.txt`
  - `training_args.bin`
  - `metrics.json`
  - 필요하면 `predictions.csv`

## 제외 항목

release zip에 넣지 않는 항목:

- `data/`
- `outputs/`
- `checkpoints/`
- `__pycache__/`
- 원본 raw data

