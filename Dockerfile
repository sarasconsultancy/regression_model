FROM python:3.9-slim
LABEL maintainer="xyz.com"

ENV PYTHONUNBUFFERED 1

COPY ./requirement_docker.txt /requirement_docker.txt
COPY ./webapp /webapp
COPY ./models/models.joblib /models/models.joblib

WORKDIR /webapp
EXPOSE 8000

RUN python -m venv /py && \
    /py/bin/pip install --upgrade pip && \
    /py/bin/pip install -r /requirement_docker.txt && \
    adduser --disabled-password --no-create-home webapp

ENV PATH="/py/bin:$PATH"

USER webapp