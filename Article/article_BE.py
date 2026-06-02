import os
import sys
import time
import json
import torch
import random
import pymysql
import feedparser
import torch.nn.functional as F
from newspaper import Article
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from dotenv import load_dotenv

# [추가] 팩트체크 모듈 경로 연결
sys.path.append(r'C:\Article\domain_fact_check')
from src import OpenAILLMEnhancer, TavilySearchAdapter, run_fact_check_service

# .env 파일에서 환경 변수 로드
load_dotenv(dotenv_path=r'C:\Article\domain_fact_check\.env')
load_dotenv(dotenv_path=r'C:\Article\.env')

# ==========================================
# 1. 설정 (DB 및 모델 경로)
# ==========================================
db_config = {
    'host': os.getenv('DB_HOST', '127.0.0.1'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'database': os.getenv('DB_NAME', 'news_db'),
    'charset': 'utf8mb4'
}

MODEL_PATH = r'C:\Article\final_model_v4'
CLICKBAIT_MODEL_PATH = r'C:\Article\klue_roberta_clickbait_title_body'
BIAS_MODEL_PATH = r'C:\Article\bias_kopolitic_transformer_3class'

MODEL_LABELS = ['정치', '경제', '사회', '문화', 'IT과학', '스포츠']
CONFIDENCE_THRESHOLD = 0.80

# ==========================================
# 2. 모델 로드 및 AI 도구 초기화
# ==========================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("⏳ AI 모델 및 팩트체크 엔진 로드 중...")
cat_tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
cat_model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH).to(device)

clickbait_tokenizer = AutoTokenizer.from_pretrained(CLICKBAIT_MODEL_PATH)
clickbait_model = AutoModelForSequenceClassification.from_pretrained(CLICKBAIT_MODEL_PATH).to(device)

bias_tokenizer = AutoTokenizer.from_pretrained(BIAS_MODEL_PATH)
bias_model = AutoModelForSequenceClassification.from_pretrained(BIAS_MODEL_PATH).to(device)

# 팩트체크용 LLM 및 검색 어댑터 설정
llm_enhancer = OpenAILLMEnhancer(model="gpt-4o-mini")
search_adapter = TavilySearchAdapter() 

print("✅ 모든 분석 엔진 준비 완료.")

# ==========================================
# 3. 데이터 처리 함수
# ==========================================
def ai_reclassify(title, content, original_category):
    input_text = f"{title} {content}"[:512]
    inputs = cat_tokenizer(input_text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = cat_model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)
    top_prob, top_idx = torch.max(probs, dim=-1)
    confidence = top_prob.item()
    model_category = MODEL_LABELS[top_idx.item()]

    if confidence >= CONFIDENCE_THRESHOLD:
        final_cat = '생활/문화' if model_category == '문화' else \
                    'IT/과학' if model_category == 'IT과학' else model_category
        return final_cat, confidence
    return original_category, confidence

def generate_verification_data(category, title, content):
    # [수정] fact_check_all 변수 추가
    bias_score, clickbait_score, fact_check_results, fact_check_all = None, None, None, None
    
    # CASE A: 정치 카테고리 - 편향성 분석
    if category == '정치':
        input_text = f"{title} {content}"
        inputs = bias_tokenizer(input_text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = bias_model(**inputs)
            prediction = torch.argmax(outputs.logits, dim=-1).item()
        bias_score = prediction
        
    # CASE B: 엔터/스포츠 - 낚시성 분석
    elif category in ['엔터', '스포츠']:
        input_text = f"{title} {content}"
        inputs = clickbait_tokenizer(input_text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = clickbait_model(**inputs)
            prediction = torch.argmax(outputs.logits, dim=-1).item()
        clickbait_score = prediction 
        
    # CASE C: 경제, 사회, IT/과학, 생활/문화 - 팩트체크 엔진 실행
    else:
        category_map = {
            '경제': 'economy',
            '사회': 'society',
            'IT/과학': 'science',
            '생활/문화': 'lifestyle_culture'
        }
        domain = category_map.get(category, 'society')

        try:
            # 외부 검색 및 LLM을 활용한 팩트체크 수행
            analysis_payload = run_fact_check_service(
                title=title,
                body=content,
                category=domain,
                search_adapter=search_adapter,
                llm_enhancer=llm_enhancer
            )
            
            # =========================================================
            # [추가] 원본 결과 전체를 JSON 문자열로 변환하여 fact_check_all에 저장
            # ensure_ascii=False를 통해 한글이 깨지지 않게 저장합니다.
            # =========================================================
            if analysis_payload:
                fact_check_all = json.dumps(analysis_payload, ensure_ascii=False)
            
            # [기존 로직 유지] 순수 텍스트 추출 및 조립
            llm_report = analysis_payload.get("report_payload", {}).get("llm_report", {})
            
            if llm_report:
                summary = llm_report.get("user_summary") or llm_report.get("overall_assessment", "팩트체크 요약 정보가 없습니다.")
                key_points = llm_report.get("key_points", [])
                
                formatted_text = f"💡 종합 요약\n{summary}\n\n📌 핵심 포인트\n"
                
                if key_points:
                    for pt in key_points:
                        formatted_text += f"• {pt}\n"
                else:
                    formatted_text += "• 요약된 포인트가 없습니다."
                    
                fact_check_results = formatted_text.strip()
            else:
                fact_check_results = "심층 분석 요약을 불러올 수 없습니다."

        except Exception as e:
            print(f"    ⚠️ 팩트체크 실패 ({category}): {e}")
            fact_check_results = "팩트체크 수행 중 오류가 발생했습니다."
            fact_check_all = None
            
    # [수정] 4개의 값을 모두 반환
    return bias_score, clickbait_score, fact_check_results, fact_check_all

def get_driver():
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    return webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)

# ==========================================
# 4. 메인 파이프라인
# ==========================================
def run_news_pipeline():
    connection = None
    driver = None
    try:
        connection = pymysql.connect(**db_config)
        cursor = connection.cursor(pymysql.cursors.DictCursor)
        driver = get_driver()

        rss_feeds = {
            '정치': 'https://news.google.com/rss/headlines/section/topic/NATION?hl=ko&gl=KR&ceid=KR:ko',
            '경제': 'https://news.google.com/rss/headlines/section/topic/BUSINESS?hl=ko&gl=KR&ceid=KR:ko',
            '사회': 'https://news.google.com/rss/search?q=%EC%82%AC%ED%9A%8C&hl=ko&gl=KR&ceid=KR:ko',
            '생활/문화': 'https://news.google.com/rss/search?q=%EC%83%9D%ED%99%9C+%EB%AC%B8%ED%99%94&hl=ko&gl=KR&ceid=KR:ko',
            'IT/과학': 'https://news.google.com/rss/headlines/section/topic/TECHNOLOGY?hl=ko&gl=KR&ceid=KR:ko',
            '엔터': 'https://news.google.com/rss/headlines/section/topic/ENTERTAINMENT?hl=ko&gl=KR&ceid=KR:ko',
            '스포츠': 'https://news.google.com/rss/headlines/section/topic/SPORTS?hl=ko&gl=KR&ceid=KR:ko'
        }

        print(f"\n📢 [뉴스 수집 시작] {time.strftime('%H:%M:%S')}")

        # [수정] 이번 실행에서 새로 INSERT된 기사 idx만 저장
        new_article_ids = []

        for category, url in rss_feeds.items():
            feed = feedparser.parse(url)
            for entry in feed.entries[:5]:
                cursor.execute("SELECT COUNT(*) as cnt FROM ARTICLE WHERE link = %s", (entry.link,))
                if cursor.fetchone()['cnt'] == 0:
                    cursor.execute("INSERT INTO ARTICLE (title, link, category) VALUES (%s, %s, %s)", 
                                   (entry.title, entry.link, category))
                    # [수정] 방금 INSERT된 기사 idx 저장
                    new_article_ids.append(cursor.lastrowid)
            connection.commit()

        # [수정] 이번 실행에서 새로 받은 기사만 분석
        if not new_article_ids:
            print("📝 이번에 새로 받은 기사가 없어 분석할 기사가 없습니다.")
            return

        placeholders = ",".join(["%s"] * len(new_article_ids))
        cursor.execute(
            f"SELECT idx, link, title, category FROM ARTICLE WHERE idx IN ({placeholders})",
            tuple(new_article_ids)
        )
        articles = cursor.fetchall()
        print(f"📝 이번 수집분 분석 대상 기사: {len(articles)}건")

        for row in articles:
            try:
                driver.get(row['link'])
                time.sleep(2)
                article = Article(driver.current_url, language='ko')
                article.download(); article.parse()
                content = article.text.strip()
                picture_url = article.top_image 
                
                if content:
                    if row['category'] in ['엔터', '스포츠']:
                        final_category = row['category']
                    else:
                        final_category, conf = ai_reclassify(row['title'], content, row['category'])

                    # [수정] fact_all 변수 추가로 받기
                    bias, click, fact_text, fact_all = generate_verification_data(final_category, row['title'], content)
                    
                    # [수정] fact_check_all 컬럼 업데이트 추가
                    sql = """UPDATE ARTICLE SET content=%s, category=%s, bias_score=%s, 
                             clickbait_score=%s, fact_check_results=%s, fact_check_all=%s, picture=%s WHERE idx=%s"""
                    cursor.execute(sql, (content, final_category, bias, click, fact_text, fact_all, picture_url, row['idx']))
                    connection.commit()
                    print(f"✅ {row['idx']}번 처리 완료 ({final_category})")
                
            except Exception as e:
                print(f"❌ {row['idx']}번 실패: {e}")
            time.sleep(1)

    finally:
        if driver: driver.quit()
        if connection: connection.close()
        print(f"🏁 파이프라인 종료. ({time.strftime('%H:%M:%S')})")

if __name__ == "__main__":
    while True:
        start_time = time.time()
        run_news_pipeline()
        wait_time = 3600 - (time.time() - start_time)
        if wait_time > 0:
            print(f"💤 {int(wait_time)}초 대기 후 다음 수집을 시작합니다.")
            time.sleep(wait_time)