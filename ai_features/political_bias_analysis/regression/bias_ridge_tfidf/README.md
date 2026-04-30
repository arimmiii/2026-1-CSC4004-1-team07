# Bias Ridge TF-IDF

정치 편향도를 1.0~5.0 연속 점수로 예측하는 Ridge 회귀 베이스라인입니다.

예시:
```bash
python ../../prepare_label1_splits.py --out-dir data/splits
python train.py --train-data data/splits/train.csv --valid-data data/splits/valid.csv --test-data data/splits/test.csv
python predict.py --model models/bias_ridge_tfidf.joblib --title "기사 제목" --content "기사 본문"
```
