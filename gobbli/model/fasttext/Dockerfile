# Build stage to compile the binary
FROM ubuntu:18.04

RUN apt-get update && apt-get install -y \
        build-essential \
        wget \
        git \
        python-dev \
        unzip \
        python-numpy \
        python-scipy \
    && rm -rf /var/cache/apk/*

WORKDIR /code

RUN git clone https://github.com/facebookresearch/fastText.git /code \
    && cd /code \
    && git checkout 5e1320a1594a026a081f8b1e5caa3085a711a625 \
    && rm -rf .git* \
    && make

# Final slim image containing just the binary
FROM ubuntu:18.04

WORKDIR /code
COPY --from=0 /code/fasttext .
ENTRYPOINT ["./fasttext"]
CMD ["help"]
