import os, tempfile
import streamlit as st
from streamlit_chat import message
from llm_rag import ChatBot

st.set_page_config(page_title="Samsung OCR ChatBot") # 앞서 정의한 PDF 임베딩 벡터 데이터베이스 RAG 모듈 임포트

def display_messages():   # 메시지 출력
    st.subheader("Chat")  
    for i, (msg, is_user) in enumerate(st.session_state["messages"]):
        message(msg, is_user=is_user, key=str(i))
    st.session_state["thinking_spinner"] = st.empty()

def process_input():   # 챗 메시지 입력 
    if st.session_state["user_input"] and len(st.session_state["user_input"].strip()) > 0:
        user_text = st.session_state["user_input"].strip()
        with st.session_state["thinking_spinner"], st.spinner(f"Thinking"):
            agent_text = st.session_state["assistant"].ask(user_text)   # 사용자 입력에서 답변 획득

        st.session_state["messages"].append((user_text, True))
        st.session_state["messages"].append((agent_text, False))

def read_and_save_file():  # file_uploader UI에서 PDF 선택 시 호출
    st.session_state["assistant"].clear() # LLM 어시스턴스 초기화
    st.session_state["messages"] = []
    st.session_state["user_input"] = ""

    for file in st.session_state["file_uploader"]:
        with tempfile.NamedTemporaryFile(delete=False) as tf:
            tf.write(file.getbuffer())
            file_path = tf.name

        with st.session_state["ingestion_spinner"], st.spinner(f"Ingesting {file.name}"):
            st.session_state["assistant"].pdf_ingest(file_path)  # 파일을 어시스턴스에 전달
        os.remove(file_path)

def page():
    if len(st.session_state) == 0:
        st.session_state["messages"] = []
        st.session_state["assistant"] = ChatBot()  # PDF, 벡터 데이터베이스, LLM 모델 호출 역할하는 객체 설정

    # UI 정의
    st.header("Samsung OCR ChatBot")  # 타이틀  
    st.subheader("Upload File")  # 서브헤더 
    st.file_uploader("Upload File", type=["pdf"],
        key="file_uploader", on_change=read_and_save_file,
        label_visibility="collapsed", accept_multiple_files=True,
    )  # 업로더 
    st.session_state["ingestion_spinner"] = st.empty()

    display_messages()  # 메시지 출력
    st.text_input("Message", key="user_input", on_change=process_input)  # 채팅 입력 버튼 생성

if __name__ == "__main__":
    page()