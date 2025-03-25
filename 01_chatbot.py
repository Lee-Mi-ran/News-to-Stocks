import gradio as gr
import chromadb
import openai
import pandas as pd
import tiktoken
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import schedule
from datetime import datetime, timedelta

# 🔹 OpenAI API 설정
openai_client = openai.OpenAI(api_key="OPEN_AI_KEY")  # OpenAI API 키 입력

# 🔹 ChromaDB 설정
# 🔹 ChromaDB 클라이언트 설정
client = chromadb.PersistentClient(path="./chroma_db_sp500")
collection = client.get_or_create_collection("sp500")

# 🔹 CSV 파일에서 데이터 읽기 및 저장 함수
def store_sp500_in_chromadb():
    df = pd.read_csv("sp500_companies.csv")  # ✅ S&P 500 기업 정보 CSV 파일 읽기
    existing_count = collection.count()

    # ✅ 기존 데이터가 있으면 저장하지 않음
    if existing_count > 0:
        print(f"✅ 기존 기업 데이터 ({existing_count}개)가 존재하여 저장하지 않습니다.")
        return

    print("🟡 S&P 500 기업 정보를 ChromaDB에 저장 중...")

    # ✅ OpenAI 임베딩 생성 함수
    def get_embedding(text):
        enc = tiktoken.encoding_for_model("text-embedding-ada-002")
        tokens = enc.encode(text)
        if len(tokens) > 7000:  # 최대 토큰 수 제한
            tokens = tokens[:7000]
            text = enc.decode(tokens)

        response = openai_client.embeddings.create(
            input=text,
            model="text-embedding-ada-002"
        )
        return response.data[0].embedding

    # ✅ 데이터 저장
    for idx, row in df.iterrows():
        company_name = row["company_name"]
        overview = row["overview"]

        # ✅ 빈 데이터는 제외
        if not overview.strip():
            continue

        # ✅ 임베딩 생성
        embedding = get_embedding(overview)

        # ✅ ChromaDB에 저장
        collection.add(
            ids=[str(idx)],
            embeddings=[embedding],
            metadatas=[{"company_name": company_name, "overview": overview}]
        )

    print("✅ S&P 500 기업 정보 저장 완료!")


news_df = None

# 🔹 뉴스 데이터 로드
# 뉴스 기사 포함 (Title, Content 컬럼 포함)
def crawl_yahoo_news():
    print("뉴스 크롤링중...")
    # ✅ Yahoo Finance Stock Market News 페이지 URL
    URL = "https://finance.yahoo.com/topic/stock-market-news/"

    # ✅ Selenium WebDriver 설정
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")  # 브라우저 창 숨김
    options.add_argument("--no-sandbox")  # 충돌 방지
    options.add_argument("--disable-dev-shm-usage")  # 메모리 문제 해결
    options.add_argument("--disable-gpu")  # GPU 가속 비활성화
    options.add_argument("user-agent=Mozilla/5.0")

    # ✅ WebDriver 실행
    driver = webdriver.Chrome(options=options)
    driver.get(URL)

    # ✅ 페이지 완전 로드 대기
    time.sleep(10)

    # ✅ 중복 방지를 위한 URL 저장
    seen_links = set()
    news_data = []

    # ✅ 기사 목록 가져오기 (최대 5개)
    try:
        articles = WebDriverWait(driver, 15).until(
            EC.presence_of_all_elements_located(
                (By.XPATH, '//div[contains(@class, "topic-stories")]//a[contains(@href, "/news/")]')
            )
        )

        for article in articles:
            link = article.get_attribute("href").strip()

            # ✅ 중복된 기사 URL 스킵
            if link in seen_links:
                continue
            seen_links.add(link)

            # 새 창에서 기사 상세 페이지 크롤링
            driver.execute_script("window.open(arguments[0]);", link)
            driver.switch_to.window(driver.window_handles[1])
            time.sleep(5)

            # ✅ 기사 제목 (cover-title)
            title = "Unknown"
            try:
                title_element = WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.CLASS_NAME, "cover-title"))
                )
                title = title_element.text.strip()
            except:
                pass

            # ✅ 언론사 (출처)
            publisher = "Unknown"
            try:
                publisher_element = driver.find_elements(By.XPATH, '//a[@class="subtle-link fin-size-small yf-1xqzjha"]')
                if publisher_element:
                    publisher = publisher_element[0].get_attribute("title").strip()
            except:
                pass

            # ✅ 날짜 크롤링 및 변환
            us_date, ko_date = "Unknown", "Unknown"
            try:
                time_element = driver.find_elements(By.CLASS_NAME, "byline-attr-meta-time")
                if time_element:
                    iso_timestamp = time_element[0].get_attribute("datetime")
                    dt_obj = datetime.strptime(iso_timestamp, "%Y-%m-%dT%H:%M:%S.%fZ")
                    us_date = dt_obj.strftime('%Y-%m-%d %H:%M')  # ✅ UTC 시간
                    ko_date = (dt_obj + timedelta(hours=9)).strftime('%Y-%m-%d %H:%M')  # ✅ 한국 시간
            except:
                pass

            # ✅ 기사 본문 (atoms-wrapper 내부 p 태그)
            content = "Unknown"
            try:
                paragraphs = driver.find_elements(By.XPATH, '//div[@class="atoms-wrapper"]//p')
                if paragraphs:
                    content = "\n".join([p.text.strip() for p in paragraphs if p.text.strip()])
            except:
                pass

            # ✅ 데이터 저장
            news_data.append({
                "Title": title,
                "URL": link,
                "Publisher": publisher,
                "US_Date": us_date,
                "KO_Date": ko_date,
                "Content": content
            })

            # 새 창 닫고 원래 창으로 복귀
            driver.close()
            driver.switch_to.window(driver.window_handles[0])

            # ✅ 5개 기사까지만 저장
            if len(news_data) >= 5:
                break

    except Exception as e:
        print(f"❌ XPath 문제 발생: {e}")

    # ✅ DataFrame 생성 후 저장
    df = pd.DataFrame(news_data)
    if df.empty:
        print("❌ 크롤링된 데이터가 없습니다. 다시 확인하세요.")
    else:
        df.to_csv("yahoo_finance_news.csv", index=False, encoding="utf-8")
        print("✅ Yahoo Finance 뉴스 크롤링 완료! ‘yahoo_finance_news.csv’ 파일 저장됨.")

    # ✅ WebDriver 종료
    driver.quit()
    store_news_in_chromadb()
    
# 🔹 토큰 수 제한 함수
def truncate_text_by_tokens(text, max_tokens=7000):
    enc = tiktoken.encoding_for_model("text-embedding-ada-002")
    tokens = enc.encode(text)
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
        text = enc.decode(tokens)
    return text

# 🔹 OpenAI 임베딩 생성
def get_embedding(text):
    text = truncate_text_by_tokens(text, max_tokens=7000)
    response = openai_client.embeddings.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return response.data[0].embedding

# 🔹 기사 번역 및 요약
def translate_and_summarize(text):
    prompt = f"""
    아래 영어 기사를 한국어로 번역 후 핵심 내용을 요약해주세요.
    공백포함 글자 수 400자 이내로 요약해주세요.
    마지막 문장이 다.라고 끝나게 요약해주세요.
    {text}
    """

    response = openai_client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500
    )

    return response.choices[0].message.content

# 🔹 유사한 기업 찾기
def find_similar_companies(content, top_n=5):
    query_embedding = get_embedding(content)

    search_result = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_n
    )

    if "metadatas" not in search_result or not search_result["metadatas"]:
        return ["⚠️ 유사한 기업을 찾을 수 없습니다. (데이터 없음)"]

    if search_result["metadatas"][0] is None:
        return ["⚠️ 유사한 기업을 찾을 수 없습니다. (NoneType 오류)"]

    best_matches = []
    for i, res in enumerate(search_result["metadatas"][0]):
        company_name = res.get("company_name", "정보 없음")
        similarity_score = search_result["distances"][0][i] if "distances" in search_result else 1.0
        similarity_percentage = round((1 - similarity_score) * 100, 2)  # 100점 만점으로 변환
        
        if isinstance(company_name, list):
            for name in company_name:
                best_matches.append(f"{name} ({similarity_percentage}%)")
        elif isinstance(company_name, str):
            best_matches.append(f"{company_name} ({similarity_percentage}%)")

    return list(set(best_matches)) if best_matches else ["⚠️ 유사한 기업을 찾을 수 없습니다."]

# ✅ **1️⃣ S&P 데이터 저장 여부 확인 후 추가**
def store_news_in_chromadb():
    global news_df
    global collection
    news_df = pd.read_csv('yahoo_finance_news.csv')
    existing_count = collection.count()
    if existing_count > 0:
        print(f"✅ 기존 데이터 ({existing_count}개)가 존재하여 저장하지 않습니다.")
        return

    print("🟡 ChromaDB 데이터 저장 중...")
    
    for idx, row in news_df.iterrows():
        title = row["Title"]
        content = row["Content"]

        if not content.strip():
            continue

        embedding = get_embedding(content)
        company_names = find_similar_companies(content)

        collection.add(
            ids=[str(idx)],
            embeddings=[embedding],
            metadatas=[{"title": title, "content": content, "company_name": company_names}]
        )
    
    print("✅ 뉴스 기사 데이터 저장 완료!")

# ✅ **2️⃣ 처음 실행 시 기사 정보 자동 출력**
def initialize_chat():
    chat_history = []

    for idx in range(len(news_df)):
        title = news_df['Title'][idx]
        link = news_df['URL'][idx]
        content = news_df['Content'][idx]

        if not content.strip():
            chat_history.append(("📢 시스템", f"🔹 {title}: ⚠️ 기사 내용 없음"))
            continue

        similar_companies = find_similar_companies(content, top_n=5)
        translated_summary = translate_and_summarize(content)

        response_message = f"""🔹 **{title}**
📌 **관련 기업**: {', '.join(similar_companies)}

📜 **한글 요약**:
{translated_summary}

**링크**
{link}
"""

        chat_history.append(("🤖 AI", response_message))

    chat_history.append(("📢 시스템", "💬 기사에 대해 궁금한 점이 있다면 질문하세요!"))

    return chat_history

# ✅ **3️⃣ 유저가 질문하면 AI가 답변**
def chatbot(user_input, chat_history):
    if user_input:
        chat_history.append((user_input, "👤 사용자"))

        similar_companies = find_similar_companies(user_input, top_n=5)
        similar_companies_text = f"관련 기업: {', '.join(similar_companies)}" if similar_companies else "관련 기업 정보 없음"
        
        prompt = f"""
        사용자 질문: {user_input}
        
        관련 기업 정보: {similar_companies_text}
        
        아래 기사 및 관련 기업 정보를 참고하여 질문에 대한 답변을 제공하세요.

        {news_df[['Title', 'Content']].to_string(index=False)}
        
        내용을 공백 포함 400자 이내로 요약해주고, 다.로 끝내주세요.
        특수기호는 빼주세요. 이모지도 빼주세요.
        """

        response = openai_client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500
        )

        chat_history.append(("🤖 AI", response.choices[0].message.content))

    return chat_history

init_history = None

def before_func():
    global init_history
    store_sp500_in_chromadb()
    crawl_yahoo_news()
    init_history = initialize_chat()
    
# ✅ Gradio 실행 (Chatbot 스타일 + 사용자 입력 지원)
def main_func():
    with gr.Blocks() as demo:
        gr.Markdown("# 📊 AI 뉴스 분석 및 기업 추천 챗봇")
        gr.Markdown("📢 뉴스 제목, 관련 기업, 요약을 먼저 제공한 후 질문할 수 있습니다.")
        
        chatbot_ui = gr.Chatbot(value=init_history, elem_id="chatbot_ui")
        user_input = gr.Textbox(label="💬 질문을 입력하세요", placeholder="기사에 대해 질문하세요...")
        submit_btn = gr.Button("질문하기")
        
        submit_btn.click(chatbot, inputs=[user_input, chatbot_ui], outputs=chatbot_ui)
    
    demo.launch()

schedule.every().hour.at(":55").do(before_func)
schedule.every().hour.at(":00").do(main_func)

# 주기적으로 작업을 실행하는 무한 루프

while True:
    schedule.run_pending()  # 예약된 작업을 실행
    time.sleep(1)  # 1초 대기


# before_func()
# main_func()
