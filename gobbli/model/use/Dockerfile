ARG GPU
FROM tensorflow/tensorflow:2.0.1${GPU:+-gpu}-py3

WORKDIR /code/use
COPY ./src/requirements.txt ./
RUN pip install -r requirements.txt

COPY ./src/ ./
