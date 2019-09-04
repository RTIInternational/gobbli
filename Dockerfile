FROM python:3.7

COPY ./setup.py /code/setup.py
COPY ./meta.json /code/meta.json
COPY ./requirements.txt /code/requirements.txt
COPY ./docs/requirements.txt /code/docs/requirements.txt

WORKDIR /code
RUN pip install --upgrade pip \
    && pip install -e '.[augment,tokenize]' \
    && pip install -r requirements.txt \
    && pip install -r docs/requirements.txt

COPY ./ /code
