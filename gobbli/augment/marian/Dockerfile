FROM pytorch/pytorch:1.3-cuda10.1-cudnn7-runtime

RUN pip install transformers==2.9.1 sentencepiece==0.1.86

COPY ./src /code/marian
WORKDIR /code/marian
