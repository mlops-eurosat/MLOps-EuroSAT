FROM python:3.12-slim AS base

WORKDIR /app

# CPU-only torch first: Cloud Run has no GPU, so the default CUDA build (~2 GB of
# unused libraries) only bloats the image, the push and the cold start. Installed
# before any COPY so this heavy layer stays cached across builds, and pinned to the
# same version as requirements_monitoring.txt so pip treats it as already satisfied.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch==2.12.1 --index-url https://download.pytorch.org/whl/cpu

# Remaining dependencies next, so source-only changes skip this install layer.
COPY requirements_monitoring.txt requirements_monitoring.txt
RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements_monitoring.txt

# Package metadata and source last: changes here only re-run the cheap install below.
# requirements.txt is the dynamic-dependency source pyproject.toml points at.
COPY pyproject.toml pyproject.toml
COPY requirements.txt requirements.txt
COPY src src/
RUN pip install . --no-deps

ENV PORT=8080
ENTRYPOINT ["sh", "-c", "uvicorn mlops_eurosat.monitoring_api:app --host 0.0.0.0 --port ${PORT:-8080}"]
