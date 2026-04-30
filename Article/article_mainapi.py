from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import Optional
import pymysql
import json
import uvicorn
import os
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

# ✅ 1. 아이디 중복 확인 전용 API 추가
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
@app.get("/news")
def get_news(category: Optional[str] = None, search: Optional[str] = None):
    conn = pymysql.connect(**db_config)
    try:
        with conn.cursor() as cursor:
            # ✅ WHERE 절에 본문이 있는 기사만 가져오는 조건 추가 (content IS NOT NULL AND content != '')
            sql = "SELECT idx, title, category, bias_score, clickbait_score FROM ARTICLE WHERE content IS NOT NULL AND content != ''"
            
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
            if res and res['fact_check_results']: res['fact_check_results'] = json.loads(res['fact_check_results'])
            return res
    finally: conn.close()

# 6. 좋아요 추가
@app.post("/like")
def like_article(data: dict):
    conn = pymysql.connect(**db_config)
    try:
        with conn.cursor() as cursor:
            cursor.execute("INSERT IGNORE INTO user_article_action (user_idx, article_idx) VALUES (%s, %s)", (data['user_idx'], data['article_idx']))
            conn.commit()
            return {"status": "success"}
    finally: conn.close()

# 7. 유저가 좋아요한 기사 목록 가져오기
@app.get("/user/{user_idx}/likes")
def get_likes(user_idx: int):
    conn = pymysql.connect(**db_config)
    try:
        with conn.cursor() as cursor:
            sql = "SELECT a.idx, a.title, a.category FROM ARTICLE a JOIN user_article_action uaa ON a.idx = uaa.article_idx WHERE uaa.user_idx = %s"
            cursor.execute(sql, (user_idx,))
            return cursor.fetchall()
    finally: conn.close()

if __name__ == "__main__":
    uvicorn.run("article_mainapi:app", host="0.0.0.0", port=8000, reload=True)
#서버 실행 코드: python -m uvicorn article_mainapi:app --reload --host 0.0.0.0