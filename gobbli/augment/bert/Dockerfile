FROM pytorch/pytorch:1.1.0-cuda10.0-cudnn7.5-runtime

RUN pip install pytorch-transformers==1.0.0

COPY ./src /code/bert
WORKDIR /code/bert
