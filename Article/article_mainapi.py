from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import Optional
import pymysql
import json
import uvicorn
import os
import random
from dotenv import load_dotenv

# .env 파일에서 환경 변수 로드
load_dotenv()

app = FastAPI()

db_config = {
    'host': os.getenv('DB_HOST', '127.0.0.1'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'database': os.getenv('DB_NAME', 'news_db'),
    'charset': 'utf8mb4',
    'cursorclass': pymysql.cursors.DictCursor
}

class UserAuth(BaseModel):
    id: str 
    password: str

# ==========================================
# [추천 시스템 설정] 알고리즘 파라미터
# ==========================================
LIKE_WEIGHT = 3                    # 좋아요 가중치 (명시적 선호 신호)
TIME_DECAY_DAYS = 7                # 시간 가중치 반감기 (7일)
EXPLORATION_RATIO = 0.1            # 탐색 비율 (10%는 비선호 카테고리에서)
BIAS_DIVERSITY_BUCKETS = 3         # 편향 점수 분할 구간 수 (좌/중/우)
DEFAULT_RECOMMEND_COUNT = 10       # 기본 추천 개수

# ==========================================
# ✅ 1. 아이디 중복 확인 전용 API 추가
# ==========================================
@app.get("/check_id/{user_id}")
def check_id(user_id: str):
    conn = pymysql.connect(**db_config)
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT idx FROM USER WHERE id = %s", (user_id,))
            if cursor.fetchone():
                return {"available": False, "message": "이미 존재하는 아이디입니다."}
            return {"available": True, "message": "사용 가능한 아이디입니다."}
    finally: conn.close()

# 2. 회원가입 (보안을 위해 여기서도 한 번 더 체크)
@app.post("/register")
def register(user: UserAuth):
    conn = pymysql.connect(**db_config)
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT idx FROM USER WHERE id = %s", (user.id,))
            if cursor.fetchone(): 
                raise HTTPException(status_code=400, detail="이미 존재하는 아이디입니다.")
            
            cursor.execute("INSERT INTO USER (id, password) VALUES (%s, %s)", (user.id, user.password))
            conn.commit()
            return {"status": "success"}
    finally: conn.close()

# 3. 로그인 (실패 시 401 반환 유지)
@app.post("/login")
def login(user: UserAuth):
    conn = pymysql.connect(**db_config)
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT idx FROM USER WHERE id = %s AND password = %s", (user.id, user.password))
            res = cursor.fetchone()
            if not res: raise HTTPException(status_code=401, detail="정보 불일치")
            return {"status": "success", "user_idx": res['idx']}
    finally: conn.close()

# 4. 뉴스 목록 가져오기
# 4. 뉴스 목록 가져오기
@app.get("/news")
def get_news(category: Optional[str] = None, search: Optional[str] = None):
    conn = pymysql.connect(**db_config)
    try:
        with conn.cursor() as cursor:
            # 💡 SELECT 부분에 picture 속성을 추가했습니다!
            sql = "SELECT idx, title, category, picture, bias_score, clickbait_score FROM ARTICLE WHERE content IS NOT NULL AND content != ''"
            
            if category and category != "전체":
                sql += f" AND category = '{category}'"
            if search:
                sql += f" AND title LIKE '%%{search}%%'"
            
            sql += " ORDER BY idx DESC"
            cursor.execute(sql)
            return cursor.fetchall()
    finally: conn.close()

# 5. 뉴스 상세 정보 가져오기
@app.get("/news/{article_id}")
def get_detail(article_id: int):
    conn = pymysql.connect(**db_config)
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT * FROM ARTICLE WHERE idx = %s", (article_id,))
            res = cursor.fetchone()
            # if res and res['fact_check_results']: res['fact_check_results'] = json.loads(res['fact_check_results'])
            return res
    finally: conn.close()

# 6. 좋아요 추가
@app.post("/like")
def like_article(data: dict):
    conn = pymysql.connect(**db_config)
    try:
        with conn.cursor() as cursor:
            cursor.execute("INSERT IGNORE INTO USER_ARTICLE_ACTION (user_idx, article_idx) VALUES (%s, %s)", (data['user_idx'], data['article_idx']))
            conn.commit()
            return {"status": "success"}
    finally: conn.close()

# 7. 유저가 좋아요한 기사 목록 가져오기
@app.get("/user/{user_idx}/likes")
def get_likes(user_idx: int):
    conn = pymysql.connect(**db_config)
    try:
        with conn.cursor() as cursor:
            # 💡 SELECT 부분에 ARTICLE.picture 속성을 추가했습니다!
            sql = "SELECT ARTICLE.idx, ARTICLE.title, ARTICLE.category, ARTICLE.picture FROM ARTICLE JOIN USER_ARTICLE_ACTION ON ARTICLE.idx = USER_ARTICLE_ACTION.article_idx WHERE USER_ARTICLE_ACTION.user_idx = %s"
            cursor.execute(sql, (user_idx,))
            return cursor.fetchall()
    finally: conn.close()


# ==========================================
# 🎯 8. 기사 추천 API (v2 - 고도화 및 프론트엔드 연동 버전)
# ==========================================

ALL_CATEGORIES = ['정치', '경제', '사회', '생활/문화', 'IT/과학', '엔터', '스포츠']

# 💡 프론트엔드가 호출하는 주소인 /recommendations 로 변경했습니다.
@app.get("/recommendations/{user_idx}")
def get_recommendations(
    user_idx: int, 
    limit: int = Query(DEFAULT_RECOMMEND_COUNT, ge=1, le=50)
):
    conn = pymysql.connect(**db_config)
    try:
        with conn.cursor() as cursor:
            # Step 0: 유저 존재 확인
            cursor.execute("SELECT idx FROM USER WHERE idx = %s", (user_idx,))
            if not cursor.fetchone():
                return [] # 유저가 없으면 빈 리스트 반환 (프론트 에러 방지)
            
            # Step 1: 카테고리 점수 계산
            cursor.execute("""
                SELECT 
                    ARTICLE.category,
                    SUM(
                        %s * EXP(-DATEDIFF(NOW(), USER_ARTICLE_ACTION.created_at) / %s)
                    ) AS score
                FROM USER_ARTICLE_ACTION
                JOIN ARTICLE ON USER_ARTICLE_ACTION.article_idx = ARTICLE.idx
                WHERE USER_ARTICLE_ACTION.user_idx = %s
                  AND USER_ARTICLE_ACTION.action_type = '좋아요'
                GROUP BY ARTICLE.category
                ORDER BY score DESC
            """, (LIKE_WEIGHT, TIME_DECAY_DAYS, user_idx))
            
            category_scores = cursor.fetchall()
            
            # Step 2: 콜드 스타트 처리 (신규 유저)
            if not category_scores:
                cursor.execute("""
                    SELECT ARTICLE.idx, ARTICLE.title, ARTICLE.category, ARTICLE.picture, ARTICLE.clickbait_score
                    FROM ARTICLE
                    LEFT JOIN USER_ARTICLE_ACTION ON ARTICLE.idx = USER_ARTICLE_ACTION.article_idx
                    WHERE ARTICLE.content IS NOT NULL AND ARTICLE.content != ''
                      AND (ARTICLE.clickbait_score IS NULL OR ARTICLE.clickbait_score = 0)
                    GROUP BY ARTICLE.idx
                    ORDER BY COUNT(USER_ARTICLE_ACTION.idx) DESC, ARTICLE.created_at DESC
                    LIMIT %s
                """, (limit,))
                # 💡 숫자(idx)만 주지 않고, 프론트엔드에서 출력할 전체 객체를 반환합니다.
                return cursor.fetchall()
            
            # Step 3: 추천 개수 분배
            exploration_count = max(1, int(limit * EXPLORATION_RATIO))
            personalized_count = limit - exploration_count
            
            total_score = sum(row['score'] for row in category_scores)
            quota = {}
            allocated = 0
            
            for row in category_scores:
                cat = row['category']
                ratio = float(row['score']) / float(total_score)
                count = int(personalized_count * ratio)
                if count > 0:
                    quota[cat] = count
                    allocated += count
            
            remaining = personalized_count - allocated
            if remaining > 0:
                top_cat = category_scores[0]['category']
                quota[top_cat] = quota.get(top_cat, 0) + remaining
            
            # Step 4: 개인화 추천
            recommended = []
            already_ids = []
            
            for cat, count in quota.items():
                articles = _fetch_articles_with_bias_diversity(
                    cursor, cat, count, user_idx, already_ids
                )
                recommended.extend(articles)
                already_ids.extend([a['idx'] for a in articles])
            
            # Step 5: 탐색 (비선호 카테고리)
            preferred_cats = set(quota.keys())
            unexplored_cats = [c for c in ALL_CATEGORIES if c not in preferred_cats]
            
            if unexplored_cats and exploration_count > 0:
                chosen_cats = random.sample(unexplored_cats, min(exploration_count, len(unexplored_cats)))
                for cat in chosen_cats:
                    articles = _fetch_articles_with_bias_diversity(
                        cursor, cat, 1, user_idx, already_ids
                    )
                    recommended.extend(articles)
                    already_ids.extend([a['idx'] for a in articles])
            
            # Step 6: 부족분 채우기
            if len(recommended) < limit:
                remaining_needed = limit - len(recommended)
                fill_articles = _fetch_fill_articles(
                    cursor, remaining_needed, user_idx, already_ids
                )
                recommended.extend(fill_articles)
            
            # 💡 숫자(idx) 배열이 아닌 기사 딕셔너리 배열 전체를 반환합니다.
            return recommended[:limit]
    
    finally:
        conn.close()


# ==========================================
# 추천 시스템 헬퍼 함수
# ==========================================

def _fetch_articles_with_bias_diversity(cursor, category, count, user_idx, exclude_ids):
    if count <= 0:
        return []
    
    buckets = BIAS_DIVERSITY_BUCKETS
    bucket_size = 100 // buckets
    
    articles = []
    per_bucket = max(1, count // buckets)
    
    exclude_placeholder = ','.join(['%s'] * len(exclude_ids)) if exclude_ids else 'NULL'
    
    for i in range(buckets):
        bias_min = i * bucket_size
        bias_max = (i + 1) * bucket_size if i < buckets - 1 else 100
        
        # 💡 SELECT 절에 ARTICLE.picture 를 추가했습니다.
        sql = f"""
            SELECT 
                ARTICLE.idx, ARTICLE.title, ARTICLE.category, ARTICLE.link, ARTICLE.picture,
                ARTICLE.bias_score, ARTICLE.clickbait_score, ARTICLE.created_at,
                COUNT(USER_ARTICLE_ACTION.idx) AS popularity
            FROM ARTICLE
            LEFT JOIN USER_ARTICLE_ACTION ON ARTICLE.idx = USER_ARTICLE_ACTION.article_idx
            WHERE ARTICLE.category = %s
              AND ARTICLE.content IS NOT NULL AND ARTICLE.content != ''
              AND (ARTICLE.clickbait_score IS NULL OR ARTICLE.clickbait_score = 0)
              AND ARTICLE.idx NOT IN (
                  SELECT article_idx FROM USER_ARTICLE_ACTION WHERE user_idx = %s
              )
              AND ARTICLE.idx NOT IN ({exclude_placeholder})
              AND ARTICLE.bias_score >= %s AND ARTICLE.bias_score <= %s
            GROUP BY ARTICLE.idx
            ORDER BY popularity DESC, ARTICLE.created_at DESC
            LIMIT %s
        """
        params = [category, user_idx] + (exclude_ids if exclude_ids else [None]) + [bias_min, bias_max, per_bucket]
        cursor.execute(sql, params)
        articles.extend(cursor.fetchall())
    
    if len(articles) < count:
        already_in_result = [a['idx'] for a in articles]
        all_excluded = exclude_ids + already_in_result
        exclude_ph = ','.join(['%s'] * len(all_excluded)) if all_excluded else 'NULL'
        
        # 💡 여기도 ARTICLE.picture 추가
        sql = f"""
            SELECT 
                ARTICLE.idx, ARTICLE.title, ARTICLE.category, ARTICLE.link, ARTICLE.picture,
                ARTICLE.bias_score, ARTICLE.clickbait_score, ARTICLE.created_at,
                COUNT(USER_ARTICLE_ACTION.idx) AS popularity
            FROM ARTICLE
            LEFT JOIN USER_ARTICLE_ACTION ON ARTICLE.idx = USER_ARTICLE_ACTION.article_idx
            WHERE ARTICLE.category = %s
              AND ARTICLE.content IS NOT NULL AND ARTICLE.content != ''
              AND (ARTICLE.clickbait_score IS NULL OR ARTICLE.clickbait_score = 0)
              AND ARTICLE.idx NOT IN (
                  SELECT article_idx FROM USER_ARTICLE_ACTION WHERE user_idx = %s
              )
              AND ARTICLE.idx NOT IN ({exclude_ph})
            GROUP BY ARTICLE.idx
            ORDER BY popularity DESC, ARTICLE.created_at DESC
            LIMIT %s
        """
        params = [category, user_idx] + (all_excluded if all_excluded else [None]) + [count - len(articles)]
        cursor.execute(sql, params)
        articles.extend(cursor.fetchall())
    
    return articles[:count]

def _fetch_fill_articles(cursor, count, user_idx, exclude_ids):
    if count <= 0:
        return []
    
    exclude_placeholder = ','.join(['%s'] * len(exclude_ids)) if exclude_ids else 'NULL'
    
    # 💡 여기도 ARTICLE.picture 추가
    sql = f"""
        SELECT 
            ARTICLE.idx, ARTICLE.title, ARTICLE.category, ARTICLE.link, ARTICLE.picture,
            ARTICLE.bias_score, ARTICLE.clickbait_score, ARTICLE.created_at,
            COUNT(USER_ARTICLE_ACTION.idx) AS popularity
        FROM ARTICLE
        LEFT JOIN USER_ARTICLE_ACTION ON ARTICLE.idx = USER_ARTICLE_ACTION.article_idx
        WHERE ARTICLE.content IS NOT NULL AND ARTICLE.content != ''
          AND (ARTICLE.clickbait_score IS NULL OR ARTICLE.clickbait_score = 0)
          AND ARTICLE.idx NOT IN (
              SELECT article_idx FROM USER_ARTICLE_ACTION WHERE user_idx = %s
          )
          AND ARTICLE.idx NOT IN ({exclude_placeholder})
        GROUP BY ARTICLE.idx
        ORDER BY popularity DESC, ARTICLE.created_at DESC
        LIMIT %s
    """
    params = [user_idx] + (exclude_ids if exclude_ids else [None]) + [count]
    cursor.execute(sql, params)
    return cursor.fetchall()


if __name__ == "__main__":
    uvicorn.run("article_mainapi:app", host="0.0.0.0", port=50000, reload=True)
#서버 실행 코드: python article_mainapi.py
#ngrok 실행 코드 : ngrok http 50000