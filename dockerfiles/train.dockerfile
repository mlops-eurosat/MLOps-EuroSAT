FROM python:3.12-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY src/ src/
COPY configs/ configs/

COPY .dvc/ .dvc/
COPY dvc.yaml dvc.yaml
COPY dvc.lock dvc.lock
COPY data/raw.dvc data/raw.dvc


# Install the CPU-only build of torch first (we run on CPU machines)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch==2.12.0 --index-url https://download.pytorch.org/whl/cpu --no-cache-dir
RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt --no-cache-dir
RUN pip install . --no-deps --no-cache-dir --verbose
RUN pip install dvc[gs] google-cloud-storage

ENTRYPOINT ["bash", "-c", "dvc pull && python -u src/mlops_eurosat/train.py"]
