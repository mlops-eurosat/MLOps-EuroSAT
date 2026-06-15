import base64
import io
import os
import tempfile
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import numpy as np
import onnxruntime as ort
from fastapi import BackgroundTasks, FastAPI, Request
from google.cloud import storage  # type: ignore[attr-defined]
from PIL import Image

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


def preprocess(image: Image.Image) -> np.ndarray:
    """Resize and normalise an image into a (1, 3, 64, 64) float32 array."""
    image = image.resize((64, 64))
    arr = np.asarray(image, dtype=np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)  # HWC -> CHW
    mean = np.array([0.3439, 0.3799, 0.4074], dtype=np.float32)
    std = np.array([0.2026, 0.1369, 0.1155], dtype=np.float32)
    arr = (arr - mean[:, None, None]) / std[:, None, None]
    return arr[None]  # add batch dim -> (1, 3, 64, 64)


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
    print(f"Loading EuroSAT ONNX model from {STORAGE_URI}")
    app.state.tmpdir = tempfile.TemporaryDirectory()
    app.state.session = _load_session_from_gcs(STORAGE_URI, app.state.tmpdir.name)
    yield
    print("Cleaning up")
    app.state.tmpdir.cleanup()
    del app.state.session


app = FastAPI(lifespan=lifespan)


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
async def health():
    return {"status": "ok"}


@app.post(PREDICT_ROUTE)
async def predict(request: Request, background_tasks: BackgroundTasks):
    body = await request.json()
    instances = body.get("instances", [])
    session = request.app.state.session

    predictions = []
    for instance in instances:
        image = _decode_instance(instance)
        x = preprocess(image)
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
