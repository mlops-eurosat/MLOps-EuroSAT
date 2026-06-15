"""Cloud Run handler for the model-registry-change automation.

Eventarc is configured to fire on the Vertex AI Cloud Audit Log entry for
``google.cloud.aiplatform.v1.ModelService.UploadModel`` and deliver it to this
service. Whenever a new model version is uploaded (i.e. a fresh ``staging``
model from training), this handler runs the performance gate and, if it passes,
promotes the model to ``production`` and deploys it to the Cloud Run API.

Run locally:  uvicorn mlops_eurosat.registry_trigger:app
Cloud Run serves it on $PORT (set automatically).
"""

import os

from fastapi import FastAPI, Request

from mlops_eurosat import model_registry

app = FastAPI()

MAX_LATENCY_S = float(os.environ.get("MAX_LATENCY_S", "5.0"))
WATCHED_METHOD = "UploadModel"


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/")
async def handle_event(request: Request):
    """Handle the Eventarc audit-log CloudEvent for a Vertex model upload."""
    try:
        body = await request.json()
    except Exception:
        body = {}

    method = body.get("protoPayload", {}).get("methodName", "")
    print(f"[registry-trigger] received event methodName={method!r}")

    if WATCHED_METHOD not in method:
        return {"status": "ignored", "method": method}

    status = model_registry.gate_promote_deploy(max_latency_s=MAX_LATENCY_S)
    print(f"[registry-trigger] result={status}")
    return {"status": status}
