FROM python:3.12-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY src/ src/
COPY data/ data/

WORKDIR /
RUN pip install -r requirements.txt --no-cache-dir --verbose
RUN pip install . --no-deps --no-cache-dir --verbose
RUN pip install dvc[gcs] google-cloud-storage

CMD ["bash", "-c", "dvc pull"]

ENTRYPOINT ["python", "-u", "src/mlops_eurosat/train.py"]
