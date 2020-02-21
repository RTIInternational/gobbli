FROM pytorch/pytorch:1.3-cuda10.1-cudnn7-devel

COPY ./src/requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

COPY ./src /code/spacy
WORKDIR /code/spacy

ARG model
RUN python -m spacy download ${model}
