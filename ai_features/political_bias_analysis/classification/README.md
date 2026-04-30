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
