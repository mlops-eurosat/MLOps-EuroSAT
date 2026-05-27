FROM python:3.12-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY src/ src/
COPY .dvc/ .dvc/
COPY data/processed.dvc data/processed.dvc

RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt --no-cache-dir
RUN pip install . --no-deps --no-cache-dir --verbose
RUN pip install dvc[gs] google-cloud-storage

ENTRYPOINT ["bash", "-c", "dvc pull && python -u src/mlops_eurosat/train.py"]
