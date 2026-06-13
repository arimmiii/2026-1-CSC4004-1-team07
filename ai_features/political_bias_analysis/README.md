# Political Bias Analysis

정치 뉴스 편향도 분석 기능용 작업 폴더입니다.

기본 구조:
- data
- models
- src

확장 구조:
- classification: `label1`을 1~5 클래스 그대로 예측
- regression: `label1`을 1.0~5.0 연속 점수로 예측

권장 실험 순서:
1. classification/bias_logreg_tfidf
2. classification/bias_svm_linear
3. classification/bias_transformer_finetune
3-1. classification/bias_transformer_kopolitic_3class
4. regression/bias_ridge_tfidf
5. regression/bias_svr_linear
6. regression/bias_transformer_regression

현재 범위:
- 사용 라벨: `label1`
- 미사용 라벨: `label2`

## CPU 학습 결과 요약

실험 데이터:
- train/valid/test는 `prepare_label1_splits.py`로 생성
- test는 `label1` 클래스 1~5 균등(각 100개)

핵심 결론:
- 분류 최선: `classification/bias_logreg_tfidf`
- 3클래스 transformer 최선: `classification/bias_transformer_kopolitic_3class`
- 회귀 최선(반올림 평가): `regression/bias_svr_linear`
- 현재 성능은 실서비스 자동 판정으로는 보수적으로 접근 필요

### 분류 모델 (label1 1~5 직접 예측)

Test 기준:
- `classification/bias_logreg_tfidf`: Accuracy `0.4620`, Macro F1 `0.4504`
- `classification/bias_svm_linear`: Accuracy `0.4360`, Macro F1 `0.4167`
- `classification/experiments/bias_svm_kernel_approx` (경량 설정): Accuracy `0.4460`, Macro F1 `0.4460`
- `classification/experiments/bias_gbdt_tfidf` (XGBoost 대체): Accuracy `0.3840`, Macro F1 `0.3434`

### 회귀 모델 (label1 연속 점수 예측)

평가 기준:
- 회귀 출력(1.0~5.0)을 반올림해 클래스(1~5)로 변환 후 Accuracy/F1 계산

Test 기준:
- `regression/bias_ridge_tfidf` (반올림): Accuracy `0.2980`, Macro F1 `0.2158`
- `regression/bias_svr_linear` (반올림): Accuracy `0.3260`, Macro F1 `0.2621`
- `regression/experiments/bias_gbdt_tfidf_regression` (XGBoost, 반올림): Accuracy `0.3100`, Macro F1 `0.2503`

### 환경/실행 참고

- `LightGBM`는 현재 환경에서 `libgomp.so.1` 이슈로 실행 실패, GBDT는 `XGBoost`로 대체해 측정
- 일부 모델은 CPU 시간 제약으로 경량 설정(`svd-components` 축소, `no-char-features`) 사용

## 실서비스 관점 메모

- 현재 수치로는 대외 사용자에게 5단계 편향도를 단정적으로 노출하기 어려움
- 내부 보조 신호로는 활용 가능
- 서비스 적용 시 권장:
  - 5클래스 대신 3구간(진보/중도/보수) 축약
  - 신뢰도 임계치 미달 시 `판단 보류` 처리

## 3클래스 개선 실험 기록

변경 목적:
- 5클래스(1~5) 직접 예측의 모호성 완화
- 실서비스 지표(`accuracy`, `macro-f1`, `top2-accuracy`) 개선

변경 내용:
- 라벨 매핑: `1-2 -> 1(liberal_block)`, `3 -> 2(neutral)`, `4-5 -> 3(conservative_block)`
- split 생성 옵션 추가: `prepare_label1_splits.py --label-mode three_class`
- 평가 스크립트 추가: `evaluate_classification.py` (`top2-accuracy` 포함)
- XGBoost 다중클래스 수 자동화 및 `softprob` 사용
- CPU 시간 제약 모델은 경량 설정 유지:
  - kernel approx: `svd-components=64`, `rbf-components=128`
  - gbdt(xgb): `--no-char-features --svd-components 128`

### 변경 전/후 결과 (Test)

| Model | Before (5-class) Accuracy | Before Macro F1 | Before Top2 | After (3-class) Accuracy | After Macro F1 | After Top2 |
|---|---:|---:|---:|---:|---:|---:|
| `classification/bias_logreg_tfidf` | 0.4620 | 0.4504 | N/A | 0.6520 | 0.6387 | 0.9200 |
| `classification/bias_svm_linear` | 0.4360 | 0.4167 | N/A | 0.6740 | 0.6627 | 0.9320 |
| `classification/experiments/bias_svm_kernel_approx` | 0.4460 | 0.4460 | N/A | 0.5720 | 0.5595 | 0.8600 |
| `classification/experiments/bias_gbdt_tfidf` (XGBoost) | 0.3840 | 0.3434 | N/A | 0.6420 | 0.6333 | 0.9060 |
| `classification/bias_transformer_kopolitic_3class` | N/A | N/A | N/A | 0.7420 | 0.7276 | 0.9280 |

메모:
- Before는 기존 5클래스 실험 로그 기준
- Before에는 `top2-accuracy`를 기록하지 않아 `N/A`
- After는 동일 데이터 소스에서 3클래스 재학습 및 재평가 결과
- `classification/bias_transformer_kopolitic_3class`는 KoPolitic 우선 + `klue/roberta-base` 폴백의 Transformer 분류 실험 결과

## Transformer 하이퍼파라미터 4실험 기록

실험 대상:
- [classification/bias_transformer_kopolitic_3class](/mnt/c/users/jaehong/desktop/sw_project/ai_features/political_bias_analysis/classification/bias_transformer_kopolitic_3class)
- backbone은 `KoPolitic` 후보 우선, 실패 시 `klue/roberta-base` 폴백
- 비교 프리셋:
  - `baseline`
  - `low_lr_longer`
  - `shorter_context_bigger_batch`
  - `stronger_regularization`

핵심 차이:
- `baseline`: 기존 기준점
- `low_lr_longer`: learning rate를 낮추고 epoch를 늘림
- `shorter_context_bigger_batch`: `max_length`를 줄이고 batch를 키워 학습 안정성 확보
- `stronger_regularization`: weight decay, warmup, label smoothing을 강화

### Transformer 4실험 결과 요약

| Preset | Valid Acc | Valid Macro F1 | Valid Top2 | Test Acc | Test Macro F1 | Test Top2 |
|---|---:|---:|---:|---:|---:|---:|
| `baseline` | 0.7622 | 0.7595 | 0.9422 | 0.7420 | 0.7276 | 0.9280 |
| `low_lr_longer` | 0.7422 | 0.7387 | 0.9422 | 0.7200 | 0.7114 | 0.9160 |
| `shorter_context_bigger_batch` | 0.7467 | 0.7416 | 0.9422 | **0.7500** | **0.7374** | 0.9160 |
| `stronger_regularization` | 0.7556 | 0.7505 | 0.9333 | 0.7420 | 0.7323 | **0.9360** |

해석:
- `shorter_context_bigger_batch`가 테스트 기준 `accuracy`, `macro-f1` 모두 최고였습니다.
- `stronger_regularization`은 테스트 `top2-accuracy`가 가장 높아, 1순위가 빗나가도 후보 2개 안에는 정답이 들어올 가능성이 가장 높았습니다.
- `baseline`은 validation 수치가 가장 높았지만, held-out test에서는 `shorter_context_bigger_batch`가 더 잘 일반화했습니다.
- `low_lr_longer`는 class 2 recall을 끌어올렸지만 precision 손실이 커서 전체 macro F1은 하락했습니다.

### Class 2 병목 비교

중도 구간(class 2)이 가장 애매한 구간이었고, 여기서 실험별 차이가 가장 선명하게 나타났습니다.

| Preset | Class 2 Precision | Class 2 Recall | Class 2 F1 |
|---|---:|---:|---:|
| `baseline` | 0.5357 | 0.7500 | 0.6250 |
| `low_lr_longer` | 0.5030 | **0.8300** | 0.6264 |
| `shorter_context_bigger_batch` | 0.5411 | 0.7900 | 0.6423 |
| `stronger_regularization` | **0.5510** | 0.8100 | **0.6559** |

해석:
- `low_lr_longer`는 중도 recall은 높았지만 precision이 크게 떨어졌습니다.
- `stronger_regularization`은 중도 precision/F1이 가장 높아, 과하게 중도로 몰아넣는 현상을 가장 잘 눌렀습니다.
- `shorter_context_bigger_batch`는 class 1, class 3까지 포함한 전체 균형이 좋아 최종 test macro F1이 최고였습니다.

### 최종 판단

- 발표/비교용 대표 모델:
  - `shorter_context_bigger_batch`
- 서비스 실험용 보조 후보:
  - `stronger_regularization`

이유:
- 대표 모델은 held-out test에서 가장 강한 일반화 성능이 필요합니다.
- 보조 후보는 top2와 class 2 안정성이 좋아, confidence 기반 후처리와 결합하기 좋습니다.

## Colab 노트북

분류 3클래스:
- [colab_bias_logreg_tfidf_3class.ipynb](/mnt/c/users/jaehong/desktop/sw_project/ai_features/political_bias_analysis/classification/bias_logreg_tfidf/colab_bias_logreg_tfidf_3class.ipynb)
- [colab_bias_svm_linear_3class.ipynb](/mnt/c/users/jaehong/desktop/sw_project/ai_features/political_bias_analysis/classification/bias_svm_linear/colab_bias_svm_linear_3class.ipynb)
- [colab_bias_svm_kernel_approx_3class.ipynb](/mnt/c/users/jaehong/desktop/sw_project/ai_features/political_bias_analysis/classification/experiments/bias_svm_kernel_approx/colab_bias_svm_kernel_approx_3class.ipynb)
- [colab_bias_gbdt_tfidf_3class.ipynb](/mnt/c/users/jaehong/desktop/sw_project/ai_features/political_bias_analysis/classification/experiments/bias_gbdt_tfidf/colab_bias_gbdt_tfidf_3class.ipynb)
- [colab_bias_kopolitic_3class.ipynb](/mnt/c/users/jaehong/desktop/sw_project/ai_features/political_bias_analysis/classification/bias_transformer_kopolitic_3class/colab_bias_kopolitic_3class.ipynb)

회귀 5클래스:
- [colab_bias_ridge_tfidf.ipynb](/mnt/c/users/jaehong/desktop/sw_project/ai_features/political_bias_analysis/regression/bias_ridge_tfidf/colab_bias_ridge_tfidf.ipynb)
- [colab_bias_svr_linear.ipynb](/mnt/c/users/jaehong/desktop/sw_project/ai_features/political_bias_analysis/regression/bias_svr_linear/colab_bias_svr_linear.ipynb)
- [colab_bias_gbdt_tfidf_regression.ipynb](/mnt/c/users/jaehong/desktop/sw_project/ai_features/political_bias_analysis/regression/experiments/bias_gbdt_tfidf_regression/colab_bias_gbdt_tfidf_regression.ipynb)
- [colab_bias_transformer_regression.ipynb](/mnt/c/users/jaehong/desktop/sw_project/ai_features/political_bias_analysis/regression/bias_transformer_regression/colab_bias_transformer_regression.ipynb)

## Release Suggestion

정치 편향도 모델 가중치는 repo 본문보다 GitHub Release로 분리하는 편이 낫습니다.

권장 release 이름:

- `bias-models-v1`

권장 asset 이름:

- `bias_svm_linear.zip`
- `bias_logreg_tfidf.zip`
- `bias_ridge_tfidf.zip`
- `bias_svr_linear.zip`
- `bias_kopolitic_3class.zip`
- `bias_transformer_regression.zip`

각 zip에는 다음만 넣으면 됩니다.

- 모델 파일
- tokenizer/config 파일
- `metrics.json`
- 필요 시 `predictions.csv`

## 시각화 산출물

발표용/보고용 차트는 [visualizations](/mnt/c/users/jaehong/desktop/sw_project/ai_features/political_bias_analysis/visualizations) 아래에 정리했습니다.

주요 파일:
- [transformer_experiments_test.svg](/mnt/c/users/jaehong/desktop/sw_project/ai_features/political_bias_analysis/visualizations/charts/transformer_experiments_test.svg)
- [transformer_experiments_valid.svg](/mnt/c/users/jaehong/desktop/sw_project/ai_features/political_bias_analysis/visualizations/charts/transformer_experiments_valid.svg)
- [transformer_experiments_class2.svg](/mnt/c/users/jaehong/desktop/sw_project/ai_features/political_bias_analysis/visualizations/charts/transformer_experiments_class2.svg)
- [transformer_experiment_summary.csv](/mnt/c/users/jaehong/desktop/sw_project/ai_features/political_bias_analysis/visualizations/data/transformer_experiment_summary.csv)
- [presentation_notes.md](/mnt/c/users/jaehong/desktop/sw_project/ai_features/political_bias_analysis/visualizations/presentation_notes.md)
