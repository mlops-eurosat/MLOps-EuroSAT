import tempfile
from contextlib import asynccontextmanager

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from fastapi import FastAPI, File, Request, UploadFile
from google.cloud import storage
from PIL import Image

from mlops_eurosat.model import Model


def preprocess(image: Image.Image) -> torch.Tensor:
    """Resize image and convert to tensor."""
    image = image.resize((64, 64))

    # Convert to float32 tensor in [0, 1]
    arr = np.asarray(image, dtype=np.float32) / 255.0

    # HWC -> CHW
    tensor = torch.from_numpy(arr).permute(2, 0, 1)

    mean = torch.tensor([0.3439, 0.3799, 0.4074])
    std = torch.tensor([0.2026, 0.1369, 0.1155])
    tensor = (tensor - mean[:, None, None]) / std[:, None, None]

    return tensor


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading EuroSAT model")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    client = storage.Client()

    bucket = client.bucket("eurosat_models")
    blob = bucket.blob("checkpoints/model.ckpt")

    with tempfile.NamedTemporaryFile(suffix=".ckpt") as f:
        blob.download_to_filename(f.name)

        checkpoint = torch.load(
            f.name,
            map_location=device,
        )

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

app = FastAPI(lifespan=lifespan)


@app.post("/predict/")
async def predict(request: Request, data: UploadFile = File(...)):
    image = Image.open(data.file).convert("RGB")

    model = request.app.state.model
    device = request.app.state.device

    x = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)[0]

    return {
        "prediction": int(torch.argmax(probs)),
        "class_name": CLASS_NAMES[int(torch.argmax(probs))],
        "probabilities": {cls: float(prob) for cls, prob in zip(CLASS_NAMES, probs.cpu().numpy())},
    }
