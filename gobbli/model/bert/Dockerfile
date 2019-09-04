ARG GPU
FROM tensorflow/tensorflow:1.11.0${GPU:+-gpu}-py3

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy modified source code in
# Base commit: d66a146741588fb208450bde15aa7db143baaa69
COPY ./src /code/bert
WORKDIR /code/bert
