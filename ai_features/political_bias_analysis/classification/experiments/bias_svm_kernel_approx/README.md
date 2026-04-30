# Bias SVM Kernel Approx

선형 SVM보다 비선형 경계를 조금 더 보려는 실험용 모델입니다.

예시:
```bash
python ../../../prepare_label1_splits.py --out-dir data/splits
python train.py --train-data data/splits/train.csv --valid-data data/splits/valid.csv --test-data data/splits/test.csv
python predict.py --model models/bias_svm_kernel_approx.joblib --title "기사 제목" --content "기사 본문"
```
