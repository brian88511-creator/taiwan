import streamlit as st
import os
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
# 🌟 V-Memory 升級：匯入新工具
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import MessagesPlaceholder # 專門用來放 "聊天記錄"
from langchain_core.messages import HumanMessage, AIMessage # 用來轉換聊天記錄的格式

# --- 1. 從 Streamlit Secrets 讀取設定 ---
# (這部分完全不變)
try:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
    os.environ["PINECONE_API_KEY"] = st.secrets["PINECONE_API_KEY"]
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
    
    PINECONE_INDEX_NAME = st.secrets["PINECONE_INDEX_NAME"]
    ACTIVE_GROQ_MODEL = st.secrets["GROQ_MODEL_NAME"]
    OPENAI_EMBED_MODEL = "text-embedding-3-small"
    
    all_keys_loaded = True
except KeyError:
    all_keys_loaded = False
    st.error("❌ 嚴重錯誤：缺少 API Keys 或設定。請在 Streamlit Cloud 的 Secrets 中設定。")
except Exception as e:
    all_keys_loaded = False
    st.error(f"❌ 發生未預期的錯誤：{e}")


# --- 2. (重要！) 🌟 V-Memory 升級：快取「對話式 RAG 鏈」 ---
# 
# 
@st.cache_resource
def get_conversational_rag_chain():
    print("... ⛔️ 正在初始化「V-Memory 對話式 RAG 系統」(只會執行一次)...")
    
    # 1. (Embeddings) 初始化「文字翻譯官」 (不變)
    embeddings = OpenAIEmbeddings(model=OPENAI_EMBED_MODEL)

    # 2. (Vector Store) 初始化 Pinecone 知識庫 (不變)
    vectorstore = PineconeVectorStore.from_existing_index(
        index_name=PINECONE_INDEX_NAME,
        embedding=embeddings
    )

    # 3. (Retriever) 建立一個「檢索器」 (不變)
    retriever = vectorstore.as_retriever(search_kwargs={'k': 3})

    # 4. (LLM) 初始化 Llama 3 大腦 (不變)
    llm = ChatGroq(model_name=ACTIVE_GROQ_MODEL)

    # --- 🌟 V-Memory 升級：我們現在需要「兩個」 Prompt ---

    # 5. (Prompt 1) 「問題改寫」提示詞 (Query Re-writing)
    # 這是 AI 的「內部工作」，用來把「那...呢？」改寫成「請介紹...」
    contextualize_q_system_prompt = """
    你是一位改寫問題的專家。
    根據「聊天記錄」和「最新的使用者問題」，
    請將「最新的使用者問題」改寫成一個「獨立、完整的問題」，
    這個新問題必須能在不知道聊天記錄的情況下被獨立理解。
    
    - 「不要」回答這個問題，你「只要」改寫它。
    - 如果「最新的使用者問題」已經是獨立完整的，就直接回傳它。
    
    【聊天記錄】:
    <chat_history>
    {chat_history}
    </chat_history>

    【最新的使用者問題】:
    {input}
    """
    
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"), # 放入聊天記錄
            ("human", "{input}"), # 放入使用者的最新問題
        ]
    )
    
    # 6. (Chain 1) 建立「歷史感知檢索器」 (History-Aware Retriever)
    # 這個鏈會 (1) 接收歷史和新問題 (2) 執行 Prompt 1 (3) 得到改寫後的問題 (4) 用新問題去檢索 Pinecone
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    # 7. (Prompt 2) 「最終回答」提示詞 (你的人設)
    # 🌟 這就是你原本的 system_prompt，但我們用新方式來組合
    # 🌟 這裡的 {context} 會由上面的 retriever 提供
    # 🌟 這裡的 {input} 是使用者「原始」的問題
    qa_system_prompt = """
    你是一位專業的台灣文化專家。
    你的任務是根據「聊天記錄」和「參考資料」來精確且專業地回答「最新的使用者問題」。

    - 請「嚴格」依照我給的「參考資料」來回答。
    - 如果「參考資料」中沒有提到，請誠實地說：「根據我目前擁有的資料，我無法回答這個問題。」
    - 你的語氣應該是親切、有深度且富含文化底蘊的。
    - 請用繁體中文回答。

    【參考資料】:
    {context}
    """

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"), # 放入聊天記錄
            ("human", "{input}"), # 放入使用者的最新問題
        ]
    )

    # 8. (Chain 2) 建立「文件組合鏈」 (Stuff Documents Chain)
    # 這個鏈會 (1) 接收所有檢索到的文件 (context) (2) 把它們 "stuff" (塞) 進 Prompt 2
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    # 9. (Chain 3) 建立「對話式 RAG 鏈」 (Conversational RAG Chain)
    # 這是最終的總鏈！
    # 它會自動執行：
    # 1. 呼叫 Chain 1 (history_aware_retriever) -> 得到改寫的問題 -> 得到文件 (context)
    # 2. 呼叫 Chain 2 (question_answer_chain) -> 傳入 (context, chat_history, input) -> 得到最終答案
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    print("🎉 V-Memory 對話式 RAG 系統已準備就緒！")
    return rag_chain


# --- 3. Streamlit 聊天介面 ---

# 🌟 V-Memory 升級：請在這裡修改你的標題、圖示、側邊欄
st.set_page_config(page_title="台灣文化專家", page_icon="🇹🇼") # 🚨 (請修改)
st.title("🇹🇼 台灣文化專家") # 🚨 (請修改)
st.caption("一個具備上下文記憶的 RAG 系統 (使用 Llama 3)")

# 範例側邊欄 (選填)
with st.sidebar:
    st.header("📖 關於這個 Demo")
    st.info(
        "這是一個使用 Conversational RAG 技術的 AI 助理。\n"
        "它能記住您之前的對話，並理解上下文。\n"
        "知識庫包含：\n"
        "- [你的論文標題 1]\n"
        "- [你的論文標題 2]\n"
    )

# 只有在所有密鑰都載入成功時，才執行
if all_keys_loaded:
    
    # 取得快取的 RAG 鏈
    try:
        conversational_rag_chain = get_conversational_rag_chain()

        # 初始化聊天記錄 (不變)
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # 顯示過去的聊天記錄 (不變)
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # 處理新的聊天輸入
        if user_question := st.chat_input("請輸入您的問題..."):
            
            # 顯示用戶訊息 (不變)
            st.session_state.messages.append({"role": "user", "content": user_question})
            with st.chat_message("user"):
                st.markdown(user_question)

            # 顯示 AI 回應
            with st.chat_message("assistant"):
                with st.spinner("AI 正在檢索並思考中..."):
                    
                    # 🌟 V-Memory 升級：準備聊天記錄 (轉換格式)
                    # 我們只傳送最後 6 則訊息 (3輪對話) 作為記憶，避免 Token 爆滿
                    chat_history_for_chain = []
                    for msg in st.session_state.messages[-6:]: # ⬅️ 只取最後 6 則
                        if msg["role"] == "user":
                            chat_history_for_chain.append(HumanMessage(content=msg["content"]))
                        elif msg["role"] == "assistant":
                            chat_history_for_chain.append(AIMessage(content=msg["content"]))
                    
                    # 🌟 V-Memory 升級：呼叫 RAG 鏈
                    # 舊版: rag_chain.invoke(user_question)
                    # 新版: 必須傳入一個包含 "input" 和 "chat_history" 的字典
                    response = conversational_rag_chain.invoke({
                        "input": user_question,
                        "chat_history": chat_history_for_chain
                    })
                    
                    # 🌟 V-Memory 升級：解析回應
                    # 舊版: response 是一個字串
                    # 新版: response 是一個字典，答案在 "answer" 鍵裡面
                    answer = response["answer"]
                    
                    st.markdown(answer)
            
            # 儲存 AI 回應 (不變)
            st.session_state.messages.append({"role": "assistant", "content": answer})

    except Exception as e:
        st.error(f"❌ 執行 RAG 鏈時發生錯誤：{e}")
        st.error("請檢查您的 Pinecone 索引名稱或 API Keys 是否正確。")
        
else:
    st.warning("App 未能初始化，請檢查 Secrets 設定。")
