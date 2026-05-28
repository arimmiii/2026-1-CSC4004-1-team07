# 2026-1-CSC4004-1-team07

# 🚀 기사리딩앱

### 👥 팀 구성
| 이름 | 역할 | 담당 업무 |
| :--- | :--- | :--- |
| 박재홍 | 팀장 | AI |
| 김정현 | 팀원 | 백엔드 |
| 김아림 | 팀원 | PM 및 프론트엔드 |
| 차영준 | 팀원 | 풀스택 |



1. 권장 환경

* Python 3.13.5 권장
  (Python 3.14 등 최신 버전은 일부 라이브러리의 빌드 에러가 발생할 수 있으므로
   안정적인 배포 및 실행을 위해 3.13.5 버전을 권장합니다.)

2. 가상환경 설정 및 패키지 설치

터미널(PowerShell)에서 아래 명령어를 순서대로 실행해 주세요.

# 가상환경 생성
python -m venv venv

# 가상환경 활성화
venv\Scripts\activate

# 필수 패키지 설치
pip install -r requirements.txt


3. 환경 변수 (API, DB) 설정

.env.example 파일을 복사하여 실제 사용할 .env 파일을 생성합니다.
생성 후 파일 내부에 개인 API 키와 DB 설정 값을 입력해 주세요.

copy .env.example .env
copy domain_fact_check\.env.example domain_fact_check\.env


4. 서버 동작 절차

로컬에서 테스트할 때는 반드시 아래 순서대로 스크립트를 실행해야 정상적으로 연동됩니다.

Step 1: ngrok 코드 실행 (외부 연결 터널링)
Step 2: article_mainapi.py 실행 (메인 API 서버)
Step 3: article_BE.py 실행 (백엔드 서버)


======================================================================
배포 및 참고 링크
======================================================================
* 배포 파일 다운로드 (Google Drive): https://drive.google.com/file/d/1zjAh6VDeJ9d8vtUDHHndGbd2QWflaQin/view?usp=drive_link