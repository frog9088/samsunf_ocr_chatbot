from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings

from langchain.schema.output_parser import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain.document_loaders import HuggingFaceDatasetLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from transformers import AutoTokenizer, pipeline
from langchain import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import VLLM
from langchain import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFacePipeline
from transformers import pipeline
import torch
from llama_cpp import Llama
import streamlit as st
from langchain.llms.base import LLM
from typing import Optional, List, Mapping, Any
from langchain.llms import LlamaCpp

USE_BGE_EMBEDDING = False

MODEL_NAME = 'ggml-model-Q5_K_M.gguf'
MODEL_PATH = "/home/asrada/바탕화면/Samsung_OCR-chatbot/models/ggml-model-Q5_K_M.gguf"
# Number of threads to use
NUM_THREADS = 8


class ChatBot:
    vector_store = None
    retriever = None
    chain = None

    def __init__(self):

        
        n_gpu_layers = -1  # The number of layers to put on the GPU. The rest will be on the CPU. If you don't know how many layers there are, you can use -1 to move all to GPU.
        n_batch = 512  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.

        # Make sure the model path is correct for your system!
        llm = LlamaCpp(
            model_path=MODEL_PATH,
            n_threads=2, # CPU cores
            n_batch=512, # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.
            n_gpu_layers=43, # Change this value based on your model and your GPU VRAM pool.
            n_ctx=4096, # Context window
)

        """
        model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1", torch_dtype=torch.float16, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")

        #model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1", load_in_4bit=True)

        # HuggingFacePipeline 객체 생성

        pipe = pipeline(
            "text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512, top_k = 3
        )
        llm = HuggingFacePipeline(pipeline=pipe)
        """
        
        """
        # Specify the model name you want to use
        model_name = "Intel/dynamic_tinybert"

        # Load the tokenizer associated with the specified model
        tokenizer = AutoTokenizer.from_pretrained(model_name, padding=True, truncation=True, max_length=512)
        """

        """
        tokenizer = AutoTokenizer.from_pretrained("yanolja/EEVE-Korean-Instruct-10.8B-v1.0")
        model = AutoModelForCausalLM.from_pretrained("yanolja/EEVE-Korean-Instruct-10.8B-v1.0")
        # Define a question-answering pipeline using the model and tokenizer
        question_answerer = pipeline(
            task="text-generation", 
            model=model, 
            tokenizer=tokenizer,
            return_tensors='pt',
        )

        # Create an instance of the HuggingFacePipeline, which wraps the question-answering pipeline
        # with additional model-specific arguments (temperature and max_length)
        llm = HuggingFacePipeline(
            pipeline=question_answerer,
            max_new_tokens = 1024,
            model_kwargs={"temperature": 0.1, "max_length": 512},
        )    
        """
        
        """
        llm = VLLM(
            model="yanolja/EEVE-Korean-Instruct-10.8B-v1.0",
            trust_remote_code=True,  # mandatory for hf models
            max_new_tokens=128,
            top_k=3,
            top_p=0.95,
            temperature=0.1
        )
        """
        #self.model = ChatOllama(model="EEVE-Korean-10.8B:latest")
        self.model = llm
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100) # PDF 텍스트 분할
        self.prompt = PromptTemplate.from_template(
            """
            <s> [INST] 당신은 금융 및 증권 관련 전문가로서 Question에 대해 Answer를 생성해야 합니다.
            반드시 다음 검색된 Context 부분을 사용하여 질문에 한국말로 답하세요.
            적절한 Context를 찾지 못한다면 모른다고 답하세요.
            최대 3개의 문장을 사용하여 간결한 답을 제공해주세요. [/INST] </s> 
            [INST] Question: {question} 
            Context: {context} 
            Answer: [/INST]
            """
        )

        if USE_BGE_EMBEDDING:
            # BGE Embedding: @Mineru
            model_name = "BAAI/bge-m3"
            # GPU Device 설정:
            # - NVidia GPU: "cuda"
            # - Mac M1, M2, M3: "mps"
            # - CPU: "cpu"
            model_kwargs = {
                "device": "cuda"
                # "device": "mps"
                # "device": "cpu"
            }
            encode_kwargs = {"normalize_embeddings": True}
            embeddings = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs,
            )
        else:
            embeddings = FastEmbedEmbeddings()

        self.embeddings = embeddings

    def pdf_ingest(self, pdf_file_path: str):
        docs = PyPDFLoader(file_path=pdf_file_path).load()  # 랭체인의 PDF 모듈 이용해 문서 로딩
        chunks = self.text_splitter.split_documents(docs)   # 문서를 청크로 분할
        chunks = filter_complex_metadata(chunks)  

        vector_store = Chroma.from_documents(documents=chunks, embedding=self.embeddings)  # 임메딩 벡터 저장소 생성 및 청크 설정
        self.retriever = vector_store.as_retriever(search_type="similarity_score_threshold",
            search_kwargs={
                "k": 3,
                "score_threshold": 0.5,
            },
        )  # 유사도 스코어 기반 벡터 검색 설정

        self.chain = ({"context": self.retriever, "question": RunnablePassthrough()} | self.prompt | self.model | StrOutputParser()) # 프롬프트 입력에 대한 모델 실행, 출력 파서 방법 설정

    def image_ingest(self, pdf_file_path: str):
        pass

    def ask(self, query: str):  # 질문 프롬프트 입력 시 호출
        if not self.chain:
            return "Please, add a PDF document first."
        return self.chain.invoke(query) 

    def clear(self):  # 초기화
        self.vector_store = None
        self.retriever = None
        self.chain = None 