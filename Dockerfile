FROM python:3.7

COPY ./setup.py ./meta.json ./requirements.txt ./README.md /code/
COPY ./docs/requirements.txt /code/docs/requirements.txt

WORKDIR /code
RUN pip install --upgrade pip \
    && pip install -e '.[augment,tokenize]' \
    && pip install -r requirements.txt \
    && pip install -r docs/requirements.txt

COPY ./ /code
