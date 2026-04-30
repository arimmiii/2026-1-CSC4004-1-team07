# Political Bias Regression

`label1`을 1.0~5.0 연속 점수로 예측하는 실험 폴더입니다.

모델:
- `bias_ridge_tfidf`: TF-IDF + Ridge Regression
- `bias_svr_linear`: TF-IDF + Linear SVR
- `bias_transformer_regression`: Transformer 회귀 헤드 파인튜닝
- `experiments/bias_gbdt_tfidf_regression`: TF-IDF/SVD + GBDT 회귀

평가 지표 권장:
- MAE
- RMSE
- Pearson/Spearman correlation

서비스 표시 예시:
- 예측 점수 `1.0~1.8`: 진보
- 예측 점수 `1.8~2.6`: 약진보
- 예측 점수 `2.6~3.4`: 중도
- 예측 점수 `3.4~4.2`: 약보수
- 예측 점수 `4.2~5.0`: 보수
