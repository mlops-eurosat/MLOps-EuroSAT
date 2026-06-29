FROM python:3.12-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements_train.txt requirements_train.txt
COPY pyproject.toml pyproject.toml
# requirements.txt is the dynamic-dependency source for `pip install .` below, so the
# build backend needs it present even though we only install the slim train set at runtime.
COPY requirements.txt requirements.txt

# Install the CPU-only build of torch first. Must match the torch pin in
# requirements_train.txt, otherwise the install below upgrades it to the default
# CUDA build from PyPI and the CPU-only optimization is silently lost.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch==2.12.1 --index-url https://download.pytorch.org/whl/cpu
RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements_train.txt
RUN --mount=type=cache,target=/root/.cache/pip pip install dvc[gs] google-cloud-storage

# Source and DVC metadata last: changes here skip the dependency layers above.
COPY src/ src/
COPY configs/ configs/
COPY .dvc/ .dvc/
COPY dvc.yaml dvc.yaml
COPY dvc.lock dvc.lock
COPY data/raw.dvc data/raw.dvc

RUN pip install . --no-deps

ENTRYPOINT ["bash", "-c", "dvc pull && python -u src/mlops_eurosat/train.py"]
