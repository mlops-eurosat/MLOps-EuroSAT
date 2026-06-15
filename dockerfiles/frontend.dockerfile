FROM python:3.12-slim

COPY requirements_frontend.txt requirements_frontend.txt
RUN pip install -r requirements_frontend.txt --no-cache-dir

COPY src/mlops_eurosat/frontend.py frontend.py

ENV PORT=8080
ENTRYPOINT ["sh", "-c", "streamlit run frontend.py --server.port=${PORT:-8080} --server.address=0.0.0.0"]
