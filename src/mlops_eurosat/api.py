"""FastAPI inference service for the EuroSAT ONNX model."""

import base64
import io
import json
import os
import tempfile
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import numpy as np
import onnxruntime as ort
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request, Response
from google.cloud import storage  # type: ignore[attr-defined]
from PIL import Image
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, Summary, generate_latest

HEALTH_ROUTE = os.environ.get("AIP_HEALTH_ROUTE", "/health")
PREDICT_ROUTE = os.environ.get("AIP_PREDICT_ROUTE", "/predict")
STORAGE_URI = os.environ.get("AIP_STORAGE_URI", "gs://eurosat_models/checkpoints")
MONITORING_BUCKET = os.environ.get("MONITORING_BUCKET", "eurosat_monitoring")

CLASS_NAMES = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
]


# Fallback for models exported before norm stats were embedded in metadata.
DEFAULT_MEAN = np.array([0.3439, 0.3799, 0.4074], dtype=np.float32)
DEFAULT_STD = np.array([0.2026, 0.1369, 0.1155], dtype=np.float32)


def preprocess(image: Image.Image, mean: np.ndarray = DEFAULT_MEAN, std: np.ndarray = DEFAULT_STD) -> np.ndarray:
    """Resize and normalise an image into the model's input format.

    Args:
        image: RGB input image of any size.
        mean: Per-channel normalisation mean.
        std: Per-channel normalisation std.

    Returns:
        Float32 array of shape ``(1, 3, 64, 64)``.
    """
    image = image.resize((64, 64))
    arr = np.asarray(image, dtype=np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)  # HWC -> CHW
    arr = (arr - mean[:, None, None]) / std[:, None, None]
    return arr[None]  # add batch dim -> (1, 3, 64, 64)


def _norm_from_session(session: ort.InferenceSession) -> tuple[np.ndarray, np.ndarray]:
    """Read normalisation mean/std from the model metadata, falling back to defaults."""
    meta = session.get_modelmeta().custom_metadata_map
    mean = np.array(json.loads(meta["norm_mean"]), dtype=np.float32) if "norm_mean" in meta else DEFAULT_MEAN
    std = np.array(json.loads(meta["norm_std"]), dtype=np.float32) if "norm_std" in meta else DEFAULT_STD
    return mean, std


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def _load_session_from_gcs(storage_uri: str, tmpdir: str) -> ort.InferenceSession:
    """Download model files from a gs:// directory into tmpdir and return an inference session."""
    assert storage_uri.startswith("gs://"), f"Expected a gs:// URI, got {storage_uri}"
    bucket_name, _, prefix = storage_uri[len("gs://") :].partition("/")
    prefix = prefix.rstrip("/")

    bucket = storage.Client().bucket(bucket_name)
    for filename in ("model.onnx", "model.onnx.data"):
        blob_path = f"{prefix}/{filename}" if prefix else filename
        blob = bucket.blob(blob_path)
        if blob.exists():
            blob.download_to_filename(f"{tmpdir}/{filename}")
            print(f"Downloaded {filename}")

    return ort.InferenceSession(f"{tmpdir}/model.onnx")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the ONNX model from GCS on startup; clean up on shutdown.

    Args:
        app: The FastAPI application; the inference session and
            normalisation stats are stored on ``app.state``.
    """
    try:
        print(f"Loading EuroSAT ONNX model from {STORAGE_URI}")
        app.state.tmpdir = tempfile.TemporaryDirectory()
        app.state.session = _load_session_from_gcs(STORAGE_URI, app.state.tmpdir.name)
        app.state.mean, app.state.std = _norm_from_session(app.state.session)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"WARNING: could not load model: {e}. /predict will return 503 until the model is available.")
        app.state.session = None
        app.state.mean, app.state.std = DEFAULT_MEAN, DEFAULT_STD
    yield
    print("Cleaning up")
    if hasattr(app.state, "tmpdir"):
        app.state.tmpdir.cleanup()
    del app.state.session


app = FastAPI(lifespan=lifespan)

prediction_requests = Counter("prediction_requests_total", "Total number of prediction requests")
prediction_errors = Counter("prediction_errors_total", "Total number of prediction errors")
prediction_latency = Histogram("prediction_latency_seconds", "Prediction latency in seconds")
instances_per_request = Summary("prediction_instances_per_request", "Number of instances per request")


@app.get("/metrics")
def metrics() -> Response:
    """Expose Prometheus metrics.

    Must answer 200 directly on /metrics: the GMP sidecar scrapes this path and
    does not follow the 307 redirect that a mounted sub-app would return.
    """
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


def _log_image(image: Image.Image, class_name: str) -> None:
    """Save raw image to GCS for offline drift analysis (background task)."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    blob_name = f"predictions/{timestamp}_{class_name}.png"
    try:
        storage.Client().bucket(MONITORING_BUCKET).blob(blob_name).upload_from_string(
            buf.getvalue(), content_type="image/png"
        )
    except Exception as e:
        print(f"[monitoring] failed to log image: {e}")


def _decode_instance(instance: dict | str) -> Image.Image:
    """Decode a Vertex prediction instance into a PIL image.

    Accepts ``{"image_b64": "<base64>"}`` or a bare base64 string.
    """
    b64 = instance["image_b64"] if isinstance(instance, dict) else instance
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")


@app.get(HEALTH_ROUTE)
async def health() -> dict:
    """Report service liveness.

    Returns:
        ``{"status": "ok"}`` while the service is running.
    """
    return {"status": "ok"}


@app.post(PREDICT_ROUTE)
async def predict(request: Request, background_tasks: BackgroundTasks) -> dict:
    """Classify base64-encoded images.

    Args:
        request: JSON body of the form ``{"instances": [{"image_b64": ...}, ...]}``.

    Returns:
        A prediction per instance: class index, class name, and per-class probabilities. 503 if no model is loaded.
    """
    prediction_requests.inc()
    with prediction_latency.time():
        try:
            body = await request.json()
            instances = body.get("instances", [])
            instances_per_request.observe(len(instances))
            session = request.app.state.session
            if session is None:
                raise HTTPException(status_code=503, detail="Model not loaded. Set AIP_STORAGE_URI and redeploy.")

            predictions = []
            for instance in instances:
                image = _decode_instance(instance)
                x = preprocess(image, request.app.state.mean, request.app.state.std)
                logits = session.run(["logits"], {"image": x})[0][0]  # (10,)
                probs = _softmax(logits)
                idx = int(np.argmax(probs))
                class_name = CLASS_NAMES[idx]

                background_tasks.add_task(_log_image, image, class_name)

                predictions.append(
                    {
                        "prediction": idx,
                        "class_name": class_name,
                        "probabilities": {cls: float(p) for cls, p in zip(CLASS_NAMES, probs)},
                    }
                )

            return {"predictions": predictions}

        except HTTPException:
            prediction_errors.inc()
            raise
        except Exception as e:
            prediction_errors.inc()
            raise HTTPException(status_code=500, detail=str(e)) from e
