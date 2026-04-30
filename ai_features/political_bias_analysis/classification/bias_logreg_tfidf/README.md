# Bias LogReg TF-IDF

정치 편향도 `label1`을 1~5 다중분류하는 베이스라인입니다.

예시:
```bash
python ../../prepare_label1_splits.py --out-dir data/splits
python train.py --train-data data/splits/train.csv --valid-data data/splits/valid.csv --test-data data/splits/test.csv
python predict.py --model models/bias_logreg_tfidf.joblib --title "기사 제목" --content "기사 본문"
```
