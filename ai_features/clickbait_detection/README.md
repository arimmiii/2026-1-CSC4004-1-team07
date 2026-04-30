# Clickbait Detection

클릭베이트/낚시성 기사 탐지 기능 관련 프로젝트를 모아둔 폴더입니다.

구성:
- `clickbait_svm_linear`: 아카이브된 선형 SVM 기준선 프로젝트
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
- 현재 메인 비교 대상은 `LogReg`와 `Transformer` 계열이며, `SVM`은 기준선 아카이브로 유지

## 결과 요약

아래 값들은 현재 폴더의 모델들을 다시 학습/평가한 최종 스냅샷입니다.

| Model | Valid Macro F1 | Test Macro F1 | Accuracy | Status |
| --- | ---: | ---: | ---: | --- |
| Linear SVM | 0.6766 | 0.6773 | 0.6773 | rerun baseline |
| Logistic Regression | 0.6790 | 0.6814 | 0.6814 | rerun reference baseline |
| KLUE RoBERTa base | 0.8011 | 0.8031 | 0.8031 | current main model |
| KLUE RoBERTa large | 0.5046 | 0.5046 | 0.5046 | rerun but not adopted |

모델별 세부 지표와 실행 방법은 각 하위 README를 따릅니다.
