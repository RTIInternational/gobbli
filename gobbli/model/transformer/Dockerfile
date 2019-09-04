FROM pytorch/pytorch:1.1.0-cuda10.0-cudnn7.5-runtime

COPY ./src/requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

COPY ./src /code/transformer
WORKDIR /code/transformer
