# Political Bias Classification

`label1`을 1~5 다중분류로 예측하는 실험 폴더입니다.

모델:
- `bias_logreg_tfidf`: TF-IDF + Logistic Regression
- `bias_svm_linear`: TF-IDF + Linear SVM
- `bias_transformer_finetune`: KLUE/KoELECTRA 계열 파인튜닝
- `bias_transformer_kopolitic_3class`: KoPolitic 우선 + KLUE RoBERTa 폴백 파인튜닝
- `experiments/bias_svm_kernel_approx`: 비선형 근사 SVM
- `experiments/bias_gbdt_tfidf`: TF-IDF/SVD + LightGBM 또는 XGBoost

평가 지표 권장:
- macro F1
- accuracy
- confusion matrix
- class-wise precision/recall/F1

## 3클래스 Transformer 4실험 비교

대상:
- [bias_transformer_kopolitic_3class](/mnt/c/users/jaehong/desktop/sw_project/ai_features/political_bias_analysis/classification/bias_transformer_kopolitic_3class)

프리셋:
- `baseline`
- `low_lr_longer`
- `shorter_context_bigger_batch`
- `stronger_regularization`

### Test 결과

| Preset | Accuracy | Macro F1 | Top2 |
|---|---:|---:|---:|
| `baseline` | 0.7420 | 0.7276 | 0.9280 |
| `low_lr_longer` | 0.7200 | 0.7114 | 0.9160 |
| `shorter_context_bigger_batch` | **0.7500** | **0.7374** | 0.9160 |
| `stronger_regularization` | 0.7420 | 0.7323 | **0.9360** |

### 해석

- `shorter_context_bigger_batch`가 held-out test 기준 가장 좋은 일반화 성능을 보였습니다.
- `stronger_regularization`은 top2와 class 2 안정성이 가장 좋았습니다.
- `baseline`은 validation 수치가 가장 높았지만 test에서는 최고가 아니었습니다.
- `low_lr_longer`는 중도(class 2) recall은 올렸지만 precision 저하가 더 컸습니다.

### Class 2 비교

| Preset | Precision | Recall | F1 |
|---|---:|---:|---:|
| `baseline` | 0.5357 | 0.7500 | 0.6250 |
| `low_lr_longer` | 0.5030 | **0.8300** | 0.6264 |
| `shorter_context_bigger_batch` | 0.5411 | 0.7900 | 0.6423 |
| `stronger_regularization` | **0.5510** | 0.8100 | **0.6559** |

시각화:
- [transformer_experiments_test.svg](/mnt/c/users/jaehong/desktop/sw_project/ai_features/political_bias_analysis/visualizations/charts/transformer_experiments_test.svg)
- [transformer_experiments_valid.svg](/mnt/c/users/jaehong/desktop/sw_project/ai_features/political_bias_analysis/visualizations/charts/transformer_experiments_valid.svg)
- [transformer_experiments_class2.svg](/mnt/c/users/jaehong/desktop/sw_project/ai_features/political_bias_analysis/visualizations/charts/transformer_experiments_class2.svg)
