# Bias GBDT TF-IDF Regression

TF-IDF + SVD 특징으로 LightGBM 또는 XGBoost 회귀를 수행하는 실험용 모델입니다.

예시:
```bash
python ../../../prepare_label1_splits.py --out-dir data/splits
python train.py --model-type lightgbm --train-data data/splits/train.csv --valid-data data/splits/valid.csv --test-data data/splits/test.csv
python predict.py --model models/bias_gbdt_tfidf_regression.joblib --title "기사 제목" --content "기사 본문"
```
