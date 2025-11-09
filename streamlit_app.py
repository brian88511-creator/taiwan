import streamlit as st
import os
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder # <<< 變更點
from langchain_core.runnables import RunnablePassthrough # 原始 import (RunnablePassthrough)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage # <<< 變更點
from langchain.chains import create_history_aware_retriever, create_retrieval_chain # <<< 變更點
from langchain.chains.combine_documents import create_stuff_documents_chain # <<< 變更點

# --- 1. 從 Streamlit Secrets 讀取設定 ---
# (此部分與您原始碼相同，保持不變)
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


# --- 2. (重要！) 快取 RAG 鏈 (已升級為對話模式) ---
# <<< 變更點：整個 get_rag_chain 函數已重構 >>>
@st.cache_resource
def get_rag_chain():
    print("... 正在初始化 RAG 系統 (對話模式)...")

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

    # 5. (System Prompt - 改寫) 建立「改寫問題」的提示詞
    # 這個 Prompt 專門用來將新問題和舊歷史結合，產生一個獨立的問題
    contextualize_q_system_prompt = """
    請根據「對話歷史」和「使用者的最新問題」，
    將「使用者的最新問題」改寫成一個「獨立、完整的問題」。
    這個問題將被用來檢索相關資料。
    如果「使用者的最新問題」本身已經很完整，就直接回傳它，不要修改。
    
    【對話歷史】:
    {chat_history}
    """
    
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"), # 放入對話歷史
            ("human", "{input}"), # 放入使用者的最新問題
        ]
    )

    # 6. (Chain - 改寫) 建立「歷史感知檢索器」鏈
    # 這個鏈會 (1) 接收歷史和新問題 (2) 呼叫 LLM 產生新問題 (3) 拿新問題去檢索
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    # 7. (System Prompt - 回答) 建立「回答問題」的提示詞 (沿用您的人設)
    # 這個 Prompt 會接收「對話歷史」、「檢索到的資料」和「新問題」
    system_prompt = """
    你是一位專業的「台灣在地文化」導覽專家與研究學者。
    你的任務是根據我提供的「參考資料」和我之前的對話，來精確且專業地回答問題。

    - 請「嚴格」依照我給的「參考資料」來回答。
    - 如果「參考資料」中沒有提到，請誠實地說：「根據我目前擁有的資料，我無法回答這個問題。」
    - 你的語氣應該是親切、有深度且富含文化底蘊的。
    - 請用繁體中文回答。

    【參考資料】:
    {context}
    """
    
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"), # 放入對話歷史
            ("human", "{input}"), # 放入使用者的最新問題
        ]
    )
    
    # 8. (Chain - 回答) 建立「文件處理鏈」
    # 這個鏈專門負責將檢索到的文件(context)塞進 Prompt 中
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    
    # 9. (Chain - 總鏈) 建立「RAG 總鏈」
    # 這是我們最終要運行的鏈
    # (1) 呼叫 history_aware_retriever (它會自動改寫問題並檢索)
    # (2) 將檢索結果和原始輸入傳遞給 question_answer_chain 來生成答案
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    print("🎉 RAG 聊天機器人 (對話模式) 已準備就緒！")
    return rag_chain

# --- 3. Streamlit 聊天介面 ---

st.title("🇹🇼 台灣文化專家")
st.caption("一個基於您在地文史資料的 RAG 系統 (使用 Llama 3 並具備記憶功能)")

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

        # <<< 變更點：修改 chat_input 的提示文字，使其與人設一致 >>>
        if user_question := st.chat_input("請輸入關於台灣在地文化的問題..."):
            # 顯示用戶訊息
            st.session_state.messages.append({"role": "user", "content": user_question})
            with st.chat_message("user"):
                st.markdown(user_question)

            # 顯示 AI 回應
            with st.chat_message("assistant"):
                with st.spinner("AI 正在檢索您的論文資料庫並思考中..."):

                    # <<< 變更點：準備 LangChain 需要的對話歷史格式 >>>
                    # st.session_state.messages 儲存的是 dict
                    # LangChain 的 MessagesPlaceholder 需要的是 HumanMessage / AIMessage 物件
                    chat_history = []
                    for msg in st.session_state.messages:
                        if msg["role"] == "user":
                            chat_history.append(HumanMessage(content=msg["content"]))
                        else:
                            chat_history.append(AIMessage(content=msg["content"]))

                    # <<< 變更點：呼叫 RAG 鏈的方式改變了 >>>
                    # 舊的：response = rag_chain.invoke(user_question)
                    # 新的：我們傳入一個 dict，包含 "input" 和 "chat_history"
                    response_dict = rag_chain.invoke(
                        {
                            "input": user_question,
                            "chat_history": chat_history 
                        }
                    )
                    
                    # <<< 變更點：從回傳的 dict 中提取答案 >>>
                    # create_retrieval_chain 的回傳是一個 dict，答案在 "answer" 欄位
                    response = response_dict["answer"]
                    
                    st.markdown(response)

            # 儲存 AI 回應
            st.session_state.messages.append({"role": "assistant", "content": response})

    except Exception as e:
        st.error(f"❌ 執行 RAG 鏈時發生錯誤：{e}")
        st.error("請檢查您的 Pinecone 索引名稱或 API Keys 是否正確。")

else:
    st.warning("App 未能初始化，請檢查 Secrets 設定。")
