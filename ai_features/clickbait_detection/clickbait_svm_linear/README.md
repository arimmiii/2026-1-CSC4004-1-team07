# Clickbait Detection with Linear SVM

`146.낚시성 기사 탐지 데이터`(`TL_*/*.json`)를 사용해 클릭베이트 이진 분류를 수행한 아카이브 프로젝트입니다.

## 데이터 및 라벨

- 원천 데이터: `data/TL_*/*.json`
- 입력 컬럼:
  - `title`: `newTitle` 우선, 없으면 `newsTitle`
  - `body`: `newsContent`
- 라벨 정의:
  - `label=1`: clickbait
  - `label=0`: nonclickbait

## 데이터 분할 방식

- 방식: `stratified reduced split (200k / 25k / 25k)`
- 공통 데이터 파일:
  - `../data/train.csv`
  - `../data/valid.csv`
  - `../data/test.csv`
- 샘플 수:
  - Train: 200,000 (clickbait 100,913 / nonclickbait 99,087)
  - Valid: 25,000 (clickbait 12,614 / nonclickbait 12,386)
  - Test: 25,000 (clickbait 12,614 / nonclickbait 12,386)

## 파이프라인

- 특징 추출: `word(1,2) TF-IDF + char(2,5) TF-IDF`
- 특징 수 제한: `word max_features=50000`, `char max_features=50000`
- 분류기: `LinearSVC(C=1.2)`

모델 파일:
- `models/linear_svm_clickbait_title_body.joblib`

## 완료/미완료 상태

- 완료:
  - 데이터 분할 생성
  - TF-IDF + LinearSVC 기준선 구축
  - 모델 저장 및 예측 스크립트 구성
- 미완료:
  - threshold 튜닝(운영 기준 정밀/재현율 트레이드오프)
  - 서비스 연동용 배치/서빙 스크립트
  - 현재 서비스 주력 모델 선정 비교 실험 반영

## 실행

```bash
py -m pip install -r requirements.txt

py train.py \
  --data ../data/train.csv \
  --valid-data ../data/valid.csv \
  --test-data ../data/test.csv \
  --title-col title \
  --body-col body \
  --label-col label \
  --model-out models/linear_svm_clickbait_title_body.joblib
```

## 예측

```bash
py predict.py \
  --model models/linear_svm_clickbait_title_body.joblib \
  --title "바뀐 제목" \
  --body "기사 본문"
```

## 개인정보/경로 노출 방지

- 공통 CSV는 `title`, `body`, `label`만 사용
- 이 README의 SVM 설정은 현재 메인 서비스 기준이 아니라 아카이브된 기준선입니다.
