import json
import os
from openai import OpenAI

# OpenAI API key should be provided via environment variable.
# Example:
#   export OPENAI_API_KEY="..."
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY environment variable is not set.")
client = OpenAI(api_key=OPENAI_API_KEY)

DATA_DIR = "data"
INPUT_FILE = os.path.join(DATA_DIR, "kist_raw.json")
OUTPUT_FILE = os.path.join(DATA_DIR, "kist_embedded.json")

def embed_data():
    if not os.path.exists(INPUT_FILE):
        print(f"❌ 에러: {INPUT_FILE} 파일을 찾을 수 없습니다. 크롤링 스크립트를 먼저 실행해 주세요.")
        return

    # 원본 데이터 로드
    print(f"📂 원본 데이터({INPUT_FILE})를 불러옵니다...")
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    embedded_results = []
    
    print(f"\n🚀 총 {len(raw_data)}개의 데이터 임베딩 시작...")
    for i, item in enumerate(raw_data):
        try:
            # 텍스트 앞 6000자 기준 안전하게 자르기
            text_to_embed = item['content'][:6000]
            
            response = client.embeddings.create(
                input=text_to_embed,
                model="text-embedding-3-small"
            )
            
            # 기존 딕셔너리에 embedding 키 추가
            item['embedding'] = response.data[0].embedding
            embedded_results.append(item)
            print(f"   [{i+1}/{len(raw_data)}] '{item['title'][:20]}...' 변환 완료")
            
        except Exception as e:
            print(f"   [{i+1}] ⚠️ 임베딩 에러 ({item['title'][:15]}...): {e}")

    # 벡터값이 포함된 최종 데이터 저장
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(embedded_results, f, ensure_ascii=False, indent=4)
        
    print(f"\n✅ 임베딩 완료! 결과가 [{OUTPUT_FILE}]에 성공적으로 저장되었습니다.")

if __name__ == "__main__":
    embed_data()
