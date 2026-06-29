FROM python:3.12-slim AS base

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY src src/
COPY requirements_monitoring.txt requirements_monitoring.txt
COPY pyproject.toml pyproject.toml

RUN pip install -r requirements_monitoring.txt --no-cache-dir
RUN pip install . --no-deps --no-cache-dir

ENV PORT=8080
ENTRYPOINT ["sh", "-c", "uvicorn mlops_eurosat.monitoring_api:app --host 0.0.0.0 --port ${PORT:-8080}"]
