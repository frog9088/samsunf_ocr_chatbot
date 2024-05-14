import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import gc
import torch
import huggingface_hub
from langchain import HuggingFacePipeline

huggingface_hub.login(token="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")

gc.collect()
torch.cuda.empty_cache()

# 허깅페이스 모델/토크나이저를 다운로드 받을 경로
# (예시)
# os.environ['HF_HOME'] = '/home/jovyan/work/tmp'
os.environ['HF_HOME'] = '/home/asrada/바탕화면/samsung_model'

from langchain import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFacePipeline
from transformers import pipeline
# HuggingFace Model ID
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1", torch_dtype=torch.float16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")

#model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1", load_in_4bit=True)

# HuggingFacePipeline 객체 생성

pipe = pipeline(
    "text-generation", model=model, tokenizer=tokenizer, max_new_tokens=100, top_k = 3
)
llm = HuggingFacePipeline(pipeline=pipe)
'''
llm = HuggingFacePipeline(
    model=model, 
    device=0,               # -1: CPU(default), 0번 부터는 CUDA 디바이스 번호 지정시 GPU 사용하여 추론
    task="text-generation", # 텍스트 생성
    model_kwargs={"temperature": 0.1, 
                  "max_length": 64}
)
'''
# 템플릿
template = """question: {question}
answer: """

# 프롬프트 템플릿 생성
prompt = PromptTemplate.from_template(template)

# LLM Chain 객체 생성
llm_chain = LLMChain(prompt=prompt, llm=llm)


question = "what is google?"
print(llm_chain.run(question=question))