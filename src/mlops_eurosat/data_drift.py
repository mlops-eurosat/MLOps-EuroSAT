"""Offline drift detection: CLIP embeddings → PCA → Evidently report.

Usage:
    python -m mlops_eurosat.data_drift \
        --bucket eurosat_monitoring \
        --train-pt data/processed/train.pt \
        --n-components 20 \
        --output reports/drift_report.html
"""

import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import typer
from evidently.legacy.metric_preset import DataDriftPreset, TargetDriftPreset
from evidently.legacy.report import Report
from google.cloud import storage
from PIL import Image
from sklearn.decomposition import PCA
from transformers import CLIPModel, CLIPProcessor

CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
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


def _load_clip(device: str) -> tuple[CLIPModel, CLIPProcessor]:
    print(f"Loading CLIP model ({CLIP_MODEL_ID})...")
    model: CLIPModel = CLIPModel.from_pretrained(CLIP_MODEL_ID)  # type: ignore[assignment]
    model = model.to(device)  # type: ignore[arg-type]
    model.eval()
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
    return model, processor


def _embed_pil_images(
    images: list[Image.Image],
    model: CLIPModel,
    processor: CLIPProcessor,
    device: str,
    batch_size: int = 32,
) -> np.ndarray:
    """Return (N, 512) CLIP image embeddings."""
    all_embeddings = []
    for i in range(0, len(images), batch_size):
        batch = images[i : i + batch_size]
        inputs = processor(images=batch, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            vision_out = model.vision_model(pixel_values=inputs["pixel_values"])
            features = model.visual_projection(vision_out.pooler_output)
        all_embeddings.append(features.cpu().numpy())
    return np.vstack(all_embeddings)


def _load_reference_images(train_pt: Path) -> tuple[list[Image.Image], list[int]]:
    """Load training images from processed .pt file."""
    print(f"Loading reference data from {train_pt}...")
    payload = torch.load(train_pt, weights_only=False)
    images_tensor = payload["images"]  # (N, 3, 64, 64), normalised
    mean = payload["mean"].view(1, 3, 1, 1)
    std = payload["std"].view(1, 3, 1, 1)
    denorm = (images_tensor * std + mean).clamp(0, 1)

    pil_images = []
    for img in denorm:
        arr = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        pil_images.append(Image.fromarray(arr))

    targets = payload["targets"].tolist()
    return pil_images, targets


def _download_prediction_images(bucket_name: str, prefix: str = "predictions/") -> tuple[list[Image.Image], list[int]]:
    """Download logged prediction PNGs from GCS, return images and predicted class indices."""
    print(f"Downloading prediction images from gs://{bucket_name}/{prefix}...")
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blobs = list(bucket.list_blobs(prefix=prefix))
    png_blobs = [b for b in blobs if b.name.endswith(".png")]

    if not png_blobs:
        raise RuntimeError(f"No prediction images found in gs://{bucket_name}/{prefix}")

    images, targets = [], []
    with tempfile.TemporaryDirectory() as tmpdir:
        for blob in png_blobs:
            local_path = Path(tmpdir) / Path(blob.name).name
            blob.download_to_filename(local_path)
            images.append(Image.open(local_path).convert("RGB"))
            # filename: <timestamp>_<ClassName>.png
            class_name = local_path.stem.split("_", 1)[-1]
            targets.append(CLASS_NAMES.index(class_name) if class_name in CLASS_NAMES else -1)

    print(f"Downloaded {len(images)} prediction images.")
    return images, targets


def data_drift(
    bucket: str = os.environ.get("MONITORING_BUCKET", "eurosat_monitoring"),
    train_pt: Path = Path("data/processed/train.pt"),
    n_components: int = 20,
    output: Path = Path("reports/drift_report.html"),
) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, processor = _load_clip(device)

    ref_images, ref_targets = _load_reference_images(train_pt)
    cur_images, cur_targets = _download_prediction_images(bucket)

    print("Extracting CLIP embeddings for reference data...")
    ref_embeddings = _embed_pil_images(ref_images, model, processor, device)

    print("Extracting CLIP embeddings for current (prediction) data...")
    cur_embeddings = _embed_pil_images(cur_images, model, processor, device)

    print(f"Fitting PCA to {n_components} components on reference embeddings...")
    pca = PCA(n_components=n_components, random_state=42)
    ref_reduced = pca.fit_transform(ref_embeddings)
    cur_reduced = pca.transform(cur_embeddings)

    col_names = [f"pc{i}" for i in range(n_components)]
    ref_df = pd.DataFrame(ref_reduced, columns=col_names)
    ref_df["target"] = ref_targets
    cur_df = pd.DataFrame(cur_reduced, columns=col_names)
    cur_df["target"] = cur_targets

    print("Running Evidently drift report...")
    report = Report(metrics=[DataDriftPreset(), TargetDriftPreset()])
    report.run(reference_data=ref_df, current_data=cur_df)

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    report.save_html(str(output))
    print(f"Drift report saved to {output}")


if __name__ == "__main__":
    typer.run(data_drift)
