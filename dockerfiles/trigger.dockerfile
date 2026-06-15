FROM python:3.12-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY src src/
COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml

RUN pip install -r requirements.txt --no-cache-dir
RUN pip install . --no-deps --no-cache-dir

ENV PORT=8080
ENTRYPOINT ["sh", "-c", "uvicorn mlops_eurosat.registry_trigger:app --host 0.0.0.0 --port ${PORT:-8080}"]
