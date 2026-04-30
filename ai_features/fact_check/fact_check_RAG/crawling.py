import requests
from bs4 import BeautifulSoup
import pdfplumber
import io
import time
import json
import os
from urllib.parse import urljoin

# 1. 초기 설정
BASE_URL = "https://kist.re.kr"
BOARD_URL = "https://kist.re.kr/ko/news/latest-research-results.do?mode=list&&articleLimit=10&article.offset={}"
DATA_DIR = "data"

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

def scrape_multiple_pages(max_pages=3):
    all_data = []
    
    for page in range(max_pages):
        offset = page * 10
        print(f"\n[수집] {page + 1}페이지 크롤링 중... (offset: {offset})")
        
        try:
            res = requests.get(BOARD_URL.format(offset), headers=headers)
            res.raise_for_status()
            soup = BeautifulSoup(res.text, 'html.parser')
            
            article_links = soup.select("a[href*='articleNo']")
            
            # [수정된 부분] 기준이 되는 주소를 정확한 게시판 경로로 맞춰줍니다.
            base_board_url = "https://kist.re.kr/ko/news/latest-research-results.do"
            unique_urls = list(set(urljoin(base_board_url, a.get('href')) for a in article_links if a.get('href')))
            
            for idx, link in enumerate(unique_urls, 1):
                try:
                    detail_res = requests.get(link, headers=headers)
                    detail_soup = BeautifulSoup(detail_res.text, 'html.parser')
                    
                    # 제목, 본문, PDF 추출
                    title_tag = detail_soup.select_one("h3.b-title-box span")
                    title = title_tag.text.strip() if title_tag else "제목 없음"
                    
                    content_tag = detail_soup.select_one("div.b-content-box")
                    fallback_text = content_tag.text.strip() if content_tag else ""
                    
                    extracted_text = ""
                    pdf_link_tag = detail_soup.select_one("div.b-file-box a[href*='.pdf']")
                    
                    if pdf_link_tag:
                        pdf_url = urljoin(link, pdf_link_tag.get('href'))
                        pdf_res = requests.get(pdf_url, headers=headers)
                        with pdfplumber.open(io.BytesIO(pdf_res.content)) as pdf:
                            for p in pdf.pages:
                                extracted_text += (p.extract_text() or "") + "\n"
                    
                    final_text = extracted_text.strip() if extracted_text.strip() else fallback_text
                    
                    # 🚨 여기서부터 수정!
                    if len(final_text) > 100:
                        all_data.append({
                            "title": title,
                            "url": link,
                            "content": final_text
                        })
                        print(f"   ({idx}/{len(unique_urls)}) ✅ 수집 완료: {title[:20]}...")
                    else:
                        # 왜 저장 안 하고 넘어갔는지 터미널에 이유를 찍어줍니다.
                        print(f"   ({idx}/{len(unique_urls)}) ❌ 패스: 텍스트가 너무 짧습니다. (길이: {len(final_text)}자)")
                        print(f"      - 기사 제목: {title}")
                    
                    time.sleep(1.2)
                    
                except Exception as e:
                    print(f"   ({idx}) 상세 페이지 에러: {e}")
                    
        except Exception as e:
            print(f"{page+1}페이지 에러: {e}")
            
    return all_data

if __name__ == "__main__":
    TARGET_PAGES = 3 # 원하는 페이지 수 입력
    print("🚀 KIST 크롤링 전용 스크립트 시작...")
    
    raw_data = scrape_multiple_pages(max_pages=TARGET_PAGES)
    
    # 순수 텍스트 원본 데이터 저장
    raw_path = os.path.join(DATA_DIR, "kist_raw.json")
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(raw_data, f, ensure_ascii=False, indent=4)
        
    print(f"\n✅ 크롤링 완료! 총 {len(raw_data)}개의 데이터가 [{raw_path}]에 저장되었습니다.")