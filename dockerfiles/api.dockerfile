FROM python:3.12-slim AS base

COPY requirements_api.txt requirements_api.txt
COPY pyproject.toml pyproject.toml

RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements_api.txt

COPY src src/
RUN pip install . --no-deps

ENV AIP_HTTP_PORT=8080
ENTRYPOINT ["sh", "-c", "uvicorn mlops_eurosat.api:app --host 0.0.0.0 --port ${AIP_HTTP_PORT:-8080}"]
