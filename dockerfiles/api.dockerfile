FROM python:3.12-slim AS base

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY src src/
COPY requirements_api.txt requirements_api.txt
COPY pyproject.toml pyproject.toml

RUN pip install -r requirements_api.txt --no-cache-dir
RUN pip install . --no-deps --no-cache-dir

ENV AIP_HTTP_PORT=8080
ENTRYPOINT ["sh", "-c", "uvicorn mlops_eurosat.api:app --host 0.0.0.0 --port ${AIP_HTTP_PORT:-8080}"]
