FROM python:3.12-slim

# Install dependencies first so this layer is cached across code-only changes.
COPY requirements_trigger.txt requirements_trigger.txt
COPY pyproject.toml pyproject.toml

RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements_trigger.txt

# Source last: editing it only invalidates the cheap package-install layer below.
COPY src src/
RUN pip install . --no-deps

ENV PORT=8080
ENTRYPOINT ["sh", "-c", "uvicorn mlops_eurosat.registry_trigger:app --host 0.0.0.0 --port ${PORT:-8080}"]
