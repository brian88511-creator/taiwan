import streamlit as st
import os
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- 1. 從 Streamlit Secrets 讀取設定 ---
# 
# 
try:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
    os.environ["PINECONE_API_KEY"] = st.secrets["PINECONE_API_KEY"]
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

    PINECONE_INDEX_NAME = st.secrets["PINECONE_INDEX_NAME"]
    ACTIVE_GROQ_MODEL = st.secrets["GROQ_MODEL_NAME"]
    OPENAI_EMBED_MODEL = "text-embedding-3-small"

    # 標記：確認所有密鑰都成功讀取
    all_keys_loaded = True

except KeyError:
    # 如果在本地測試（沒有 st.secrets），這會發生
    # 為了讓 App 至少能顯示錯誤，我們捕捉它
    all_keys_loaded = False
    st.error("❌ 嚴重錯誤：缺少 API Keys 或設定。請在 Streamlit Cloud 的 Secrets 中設定。")
except Exception as e:
    all_keys_loaded = False
    st.error(f"❌ 發生未預期的錯誤：{e}")


# --- 2. (重要！) 快取 RAG 鏈 ---
# 使用 @st.cache_resource 確保模型和鏈只被載入一次
# 這樣 App 才有效率，不會每次用戶提問都重新載入
@st.cache_resource
def get_rag_chain():
    print("... 正在初始化 RAG 系統 (只會執行一次)...")

    # 1. (Embeddings) 初始化「文字翻譯官」
    embeddings = OpenAIEmbeddings(model=OPENAI_EMBED_MODEL)

    # 2. (Vector Store) 初始化 Pinecone 知識庫
    vectorstore = PineconeVectorStore.from_existing_index(
        index_name=PINECONE_INDEX_NAME,
        embedding=embeddings
    )

    # 3. (Retriever) 建立一個「檢索器」
    retriever = vectorstore.as_retriever(search_kwargs={'k': 3}) # k=3: 每次抓 3 份相關資料

    # 4. (LLM) 初始化 Llama 3 大腦
    llm = ChatGroq(model_name=ACTIVE_GROQ_MODEL)

    # 5. (System Prompt) 建立「系統提示詞」(你的人設)
    system_prompt = """
    你是一位專業的「台灣在地文化」導覽專家與研究學者。
你的任務是參考我提供的「參考資料」結合你原本的知識來回答問題。

- 你的語氣應該是親切、有深度且富含文化底蘊的。
- 請主要根據使用者使用的語言回答。

    【參考資料】:
    {context}
    """

    # 6. (Chain) 建立完整的 RAG 處理鏈
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{question}"),
        ]
    )

    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    print("🎉 RAG 聊天機器人已準備就緒！")
    return rag_chain

# --- 3. Streamlit 聊天介面 ---

st.title("🇹🇼 台灣文化專家")
st.caption("一個基於您在地文史資料的 RAG 系統 (使用 Llama 3)")

# 只有在所有密鑰都載入成功時，才執行 RAG 鏈和聊天
if all_keys_loaded:

    # 取得快取的 RAG 鏈
    try:
        rag_chain = get_rag_chain()

        # 初始化聊天記錄 (儲存在 session_state)
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # 顯示過去的聊天記錄
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # 處理新的聊天輸入
        if user_question := st.chat_input("請輸入關於台灣文化的問題..."):
            # 顯示用戶訊息
            st.session_state.messages.append({"role": "user", "content": user_question})
            with st.chat_message("user"):
                st.markdown(user_question)

            # 顯示 AI 回應
            with st.chat_message("assistant"):
                with st.spinner("AI 正在檢索您的論文資料庫並思考中..."):

                    # 執行 RAG 鏈
                    response = rag_chain.invoke(user_question)
                    st.markdown(response)

            # 儲存 AI 回應
            st.session_state.messages.append({"role": "assistant", "content": response})

    except Exception as e:
        st.error(f"❌ 執行 RAG 鏈時發生錯誤：{e}")
        st.error("請檢查您的 Pinecone 索引名稱或 API Keys 是否正確。")

else:
    st.warning("App 未能初始化，請檢查 Secrets 設定。")
