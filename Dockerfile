# base img
FROM nvidia/cuda:12.1.0-base-ubuntu20.04

# timezone 설정 : Asia/Seoul
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul

# apt-get update
RUN apt-get update \
&& apt-get install -y wget git vim g++ gcc make curl locales \
&& rm -rf /var/lib/apt/lists/*

# miniconda 설치
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
RUN bash ~/miniconda.sh -b -p $HOME/miniconda
ENV PATH="/root/miniconda/bin:${PATH}"

# create & activate conda env
RUN conda create -n samsung python=3.10 -y && conda init bash
RUN echo "conda activate samsung" >> ~/.bashrc
SHELL ["/bin/bash", "--login", "-c"]

# github clone해오기
WORKDIR /Samsung_OCR-Chatbot
RUN git clone -b ollama https://github.com/namkidong98/Samsung_OCR-Chatbot.git .
RUN cd /Samsung_OCR-Chatbot

# git clone해온 디렉토리에 있는 requirements.txt로 설치
RUN /root/miniconda/envs/samsung/bin/pip install -r requirements.txt

# ollama 설치
RUN curl https://ollama.ai/install.sh | sh

# huggingface에서 model 설치
# RUN /root/miniconda/envs/samsung/bin/pip install huggingface-cli
# RUN huggingface-cli download heegyu/EEVE-Korean-Instruct-10.8B-v1.0-GGUF ggml-model-Q5_K_M.gguf --local-dir /models --local-dir-use-symlinks False
RUN wget -O models/ggml-model-Q5_K_M.gguf https://huggingface.co/heegyu/EEVE-Korean-Instruct-10.8B-v1.0-GGUF/resolve/main/ggml-model-Q5_K_M.gguf?download=true

# Modelfile로 만들기
RUN ollama list
RUN ollama create EEVE-Korean-10.8B -f models/Modelfile

# 마지막으로 streamlit 실행
CMD ["streamlit", "run", "app.py"]