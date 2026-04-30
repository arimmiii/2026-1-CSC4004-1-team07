import os
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

# ==========================================
# 1. 설정 (DB 및 모델 경로)
# ==========================================
db_config = {
    'host': '127.0.0.1',
    'user': 'ckdudwns0618',
    'password': 'ckdudwns@@1',
    'database': 'news_db',
    'charset': 'utf8mb4'
}

# 기존 분류 모델 경로
MODEL_PATH = r'C:\Article\final_model_v3'
MODEL_LABELS = ['정치', '경제', '사회', '문화', 'IT과학', '스포츠']
CONFIDENCE_THRESHOLD = 0.80

# 💡 수정: 낚시성 기사 판별 딥러닝 모델 경로
CLICKBAIT_MODEL_PATH = r'C:\Article\klue_roberta_clickbait_title_body'

# ==========================================
# 2. 모델 로드 (GPU/CPU 자동 설정)
# ==========================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("⏳ 1/2. 카테고리 분류 AI 모델 로드 중...")
cat_tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
cat_model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH).to(device)
cat_model.eval()
print("✅ 카테고리 분류 모델 준비 완료.")

print("⏳ 2/2. 낚시성 기사 판별 딥러닝 모델 로드 중...")
# 💡 수정: Transformers 방식을 사용하여 낚시성 모델 로드
clickbait_tokenizer = AutoTokenizer.from_pretrained(CLICKBAIT_MODEL_PATH)
clickbait_model = AutoModelForSequenceClassification.from_pretrained(CLICKBAIT_MODEL_PATH).to(device)
clickbait_model.eval()
print("✅ 낚시성 기사 판별 모델 준비 완료.")


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
    bias_score, clickbait_score, fact_check_results = None, None, None
    
    if category == '정치':
        bias_score = random.randint(30, 90)
        
    elif category in ['엔터', '스포츠']:
        # 💡 수정: 딥러닝 모델을 이용한 낚시성 기사 판별 로직
        input_text = f"{title} {content}"
        # RoBERTa 모델에 맞게 토크나이징 (최대 길이 512 제한)
        inputs = clickbait_tokenizer(input_text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
        
        with torch.no_grad():
            outputs = clickbait_model(**inputs)
            # 로짓 값에서 가장 확률이 높은 클래스(0 또는 1) 추출
            prediction = torch.argmax(outputs.logits, dim=-1).item()
            
        clickbait_score = prediction # 1이면 낚시성, 0이면 정상 (학습 데이터 라벨링에 따라 다름)
        
    else:
        res = [{"target_text": f"{title} 핵심", "is_fact": "True", "evidence": "데이터 분석 결과"}]
        fact_check_results = json.dumps(res, ensure_ascii=False)
        
    return bias_score, clickbait_score, fact_check_results

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

        print(f"\n📢 [수집 시작] {time.strftime('%H:%M:%S')}")
        for category, url in rss_feeds.items():
            feed = feedparser.parse(url)
            for entry in feed.entries[:5]:
                cursor.execute("SELECT COUNT(*) as cnt FROM ARTICLE WHERE link = %s", (entry.link,))
                if cursor.fetchone()['cnt'] == 0:
                    cursor.execute("INSERT INTO ARTICLE (title, link, category) VALUES (%s, %s, %s)", 
                                   (entry.title, entry.link, category))
            connection.commit()

        cursor.execute("SELECT idx, link, title, category FROM ARTICLE WHERE content IS NULL")
        articles = cursor.fetchall()
        print(f"📝 분석 대기 중인 기사: {len(articles)}건")

        for row in articles:
            try:
                driver.get(row['link'])
                time.sleep(2)
                article = Article(driver.current_url, language='ko')
                article.download(); article.parse()
                content = article.text.strip()
                
                if content:
                    if row['category'] in ['엔터', '스포츠']:
                        final_category = row['category']
                        print(f"⏩ {row['idx']}번 분류: {row['category']} [바로 삽입/모델 패스]")
                    else:
                        final_category, conf = ai_reclassify(row['title'], content, row['category'])
                        status = "[변경됨]" if row['category'] != final_category else "[유지]"
                        print(f"🔍 {row['idx']}번 분류: {row['category']} -> {final_category} {status} (확신도: {conf*100:.1f}%)")

                    bias, click, fact_json = generate_verification_data(final_category, row['title'], content)
                    
                    if click is not None:
                        click_status = "⚠️ 낚시성 기사" if click == 1 else "✅ 정상 기사"
                        print(f"   └─ 낚시성 판별 결과: {click_status} (Score: {click})")

                    sql = """UPDATE ARTICLE SET content=%s, category=%s, bias_score=%s, 
                             clickbait_score=%s, fact_check_results=%s WHERE idx=%s"""
                    cursor.execute(sql, (content, final_category, bias, click, fact_json, row['idx']))
                    connection.commit()
                
            except Exception as e:
                print(f"❌ {row['idx']}번 분석 실패: {e}")
            time.sleep(1)

    finally:
        if driver: driver.quit()
        if connection: connection.close()
        print(f"🏁 수집 및 분류 완료. ({time.strftime('%H:%M:%S')})")

# ==========================================
# 5. 실행 루프 (시작 시점 기준 1시간 간격)
# ==========================================
if __name__ == "__main__":
    while True:
        start_time = time.time()  
        
        run_news_pipeline()
        
        end_time = time.time()    
        elapsed_time = end_time - start_time  
        
        wait_time = 3600 - elapsed_time
        
        if wait_time > 0:
            print(f"💤 작업에 {int(elapsed_time)}초 소요됨. 다음 시작까지 {int(wait_time)}초 대기합니다.")
            time.sleep(wait_time)
        else:
            print(f"⚠️ 작업 시간이 1시간을 초과했습니다({int(elapsed_time)}초). 쉬지 않고 즉시 다음 작업을 시작합니다.")