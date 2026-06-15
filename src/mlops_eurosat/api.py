import base64
import io
import os
import tempfile
from contextlib import asynccontextmanager

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, Request
from google.cloud import storage  # type: ignore[attr-defined]
from PIL import Image

HEALTH_ROUTE = os.environ.get("AIP_HEALTH_ROUTE", "/health")
PREDICT_ROUTE = os.environ.get("AIP_PREDICT_ROUTE", "/predict")
STORAGE_URI = os.environ.get("AIP_STORAGE_URI", "gs://eurosat_models/checkpoints")

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


def _load_session_from_gcs(storage_uri: str) -> ort.InferenceSession:
    """Download model.onnx from a gs:// directory and return an inference session."""
    assert storage_uri.startswith("gs://"), f"Expected a gs:// URI, got {storage_uri}"
    bucket_name, _, prefix = storage_uri[len("gs://") :].partition("/")
    blob_path = f"{prefix.rstrip('/')}/model.onnx" if prefix else "model.onnx"

    blob = storage.Client().bucket(bucket_name).blob(blob_path)
    with tempfile.NamedTemporaryFile(suffix=".onnx") as f:
        blob.download_to_filename(f.name)
        return ort.InferenceSession(f.name)


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"Loading EuroSAT ONNX model from {STORAGE_URI}")
    app.state.session = _load_session_from_gcs(STORAGE_URI)
    yield
    print("Cleaning up")
    del app.state.session


app = FastAPI(lifespan=lifespan)


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
async def predict(request: Request):
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
        predictions.append(
            {
                "prediction": idx,
                "class_name": CLASS_NAMES[idx],
                "probabilities": {cls: float(p) for cls, p in zip(CLASS_NAMES, probs)},
            }
        )

    return {"predictions": predictions}
