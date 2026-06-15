FROM python:3.12-slim AS base

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY src src/
COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml

# Install the CPU-only build of torch first (serving runs on CPU)
RUN pip install torch==2.12.0 --index-url https://download.pytorch.org/whl/cpu --no-cache-dir
RUN pip install -r requirements.txt --no-cache-dir --verbose
RUN pip install . --no-deps --no-cache-dir --verbose

ENV AIP_HTTP_PORT=8080
ENTRYPOINT ["sh", "-c", "uvicorn mlops_eurosat.api:app --host 0.0.0.0 --port ${AIP_HTTP_PORT:-8080}"]
