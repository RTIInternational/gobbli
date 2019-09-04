FROM python:3.7

# Copy essentials in to install requirements
COPY ./setup.py /code/setup.py
COPY ./meta.json /code/meta.json
COPY ./requirements.txt /code/requirements.txt
COPY ./benchmark/requirements.txt /code/benchmark/requirements.txt

# Install dependencies
WORKDIR /code
RUN pip install -e '.[augment,tokenize]' \
    && pip install -r requirements.txt \
    && pip install -r benchmark/requirements.txt

# Copy the rest of the repository in
COPY ./ /code

EXPOSE 8080
CMD [ \
    "jupyter", "lab", \
    "--ip", "0.0.0.0", \
    "--allow-root", \
    "--port", "8080", \
    "--no-browser", \
    "--LabApp.token=''" \
]
