FROM python:3.12-slim AS base

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY src src/
COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml

RUN pip install -r requirements.txt --no-cache-dir --verbose
RUN pip install . --no-deps --no-cache-dir --verbose

ENV AIP_HTTP_PORT=8080
ENTRYPOINT ["sh", "-c", "uvicorn mlops_eurosat.api:app --host 0.0.0.0 --port ${AIP_HTTP_PORT:-8080}"]
