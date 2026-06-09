import base64
import io
import os
import tempfile
from contextlib import asynccontextmanager

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from fastapi import FastAPI, Request
from google.cloud import storage  # type: ignore[attr-defined]
from PIL import Image

from mlops_eurosat.model import Model

HEALTH_ROUTE = os.environ.get("AIP_HEALTH_ROUTE", "/health")
PREDICT_ROUTE = os.environ.get("AIP_PREDICT_ROUTE", "/predict")
STORAGE_URI = os.environ.get("AIP_STORAGE_URI", "gs://eurosat_models/checkpoints")
CHECKPOINT_BLOB = "model.ckpt"

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


def preprocess(image: Image.Image) -> torch.Tensor:
    """Resize image and convert to a normalised CHW tensor."""
    image = image.resize((64, 64))

    arr = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1)  # HWC -> CHW

    mean = torch.tensor([0.3439, 0.3799, 0.4074])
    std = torch.tensor([0.2026, 0.1369, 0.1155])
    tensor = (tensor - mean[:, None, None]) / std[:, None, None]
    return tensor


def _load_checkpoint_from_gcs(storage_uri: str, device: torch.device) -> dict:
    """Download the checkpoint from a gs:// directory and load it onto `device`."""
    assert storage_uri.startswith("gs://"), f"Expected a gs:// URI, got {storage_uri}"
    bucket_name, _, prefix = storage_uri[len("gs://") :].partition("/")
    blob_path = f"{prefix.rstrip('/')}/{CHECKPOINT_BLOB}" if prefix else CHECKPOINT_BLOB

    client = storage.Client()
    blob = client.bucket(bucket_name).blob(blob_path)
    with tempfile.NamedTemporaryFile(suffix=".ckpt") as f:
        blob.download_to_filename(f.name)
        return torch.load(f.name, map_location=device)


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"Loading EuroSAT model from {STORAGE_URI}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = _load_checkpoint_from_gcs(STORAGE_URI, device)
    model = Model()
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()

    app.state.device = device
    app.state.model = model
    yield

    print("Cleaning up")
    del app.state.model
    del app.state.device


app = FastAPI(lifespan=lifespan)


def _decode_instance(instance: dict | str) -> Image.Image:
    """Decode a single Vertex prediction instance into a PIL image.

    Accepts ``{"image_b64": "<base64>"}`` (or a bare base64 string) holding the
    raw bytes of a JPEG/PNG image.
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

    model = request.app.state.model
    device = request.app.state.device

    predictions = []
    for instance in instances:
        image = _decode_instance(instance)
        x = preprocess(image).unsqueeze(0).to(device)

        with torch.no_grad():
            probs = F.softmax(model(x), dim=1)[0]

        idx = int(torch.argmax(probs))
        predictions.append(
            {
                "prediction": idx,
                "class_name": CLASS_NAMES[idx],
                "probabilities": {cls: float(p) for cls, p in zip(CLASS_NAMES, probs.cpu().numpy())},
            }
        )

    return {"predictions": predictions}
