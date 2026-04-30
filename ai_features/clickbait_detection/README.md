# Clickbait Detection

클릭베이트/낚시성 기사 탐지 기능 관련 프로젝트를 모아둔 폴더입니다.

구성:
- `clickbait_svm_linear`: 현재 최고 성능의 선형 SVM 프로젝트
- `clickbait_logreg_tfidf`: TF-IDF + Logistic Regression 비교 모델
- `clickbait_transformer_finetune`: KLUE-RoBERTa 파인튜닝 프로젝트
- `experiments`: 성능이 낮았거나 미완료인 실험 폴더

데이터 기준:
- 공통 데이터 CSV는 `data/{train,valid,test}.csv`
- 컬럼 구성은 `title`, `body`, `label`
- 세 프로젝트 모두 이 공통 데이터를 재사용
- Colab 데이터 확인 노트북: `colab_title_body_from_archives_pipeline.ipynb`
- 원본 `raw data`에서 reduced split 생성: `make_reduced_splits.py`
- 현재 크기: Train `200,000`, Valid `25,000`, Test `25,000`
