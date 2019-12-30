FROM pytorch/pytorch:1.3-cuda10.1-cudnn7-runtime

RUN pip install transformers==2.3.0

COPY ./src /code/bert
WORKDIR /code/bert
