"""Monitoring API serving Evidently drift reports on demand.

Compares recent prediction images against reference CLIP embeddings, either
on request (``/report``) or gated for Cloud Scheduler (``/check``).
"""

import json
import os
import tempfile
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from evidently.legacy.metric_preset import DataDriftPreset, TargetDriftPreset
from evidently.legacy.report import Report
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from google.cloud import storage
from PIL import Image
from sklearn.decomposition import PCA
from transformers import CLIPModel, CLIPProcessor

MONITORING_BUCKET = os.environ.get("MONITORING_BUCKET", "eurosat_monitoring")
CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
N_COMPONENTS = 20
MIN_SAMPLES = int(os.environ.get("MIN_SAMPLES", "100"))
MAX_STALENESS_HOURS = int(os.environ.get("MAX_STALENESS_HOURS", "24"))
STATE_BLOB = "state/last_run.json"

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


# ---------------------------------------------------------------------------
# GCS state helpers
# ---------------------------------------------------------------------------


def _read_state(bucket_name: str) -> dict:
    """Read last_run state from GCS. Returns defaults if not found."""
    try:
        blob = storage.Client().bucket(bucket_name).blob(STATE_BLOB)
        return json.loads(blob.download_as_text())
    except Exception:
        return {"last_run_at": None, "processed_count": 0}


def _write_state(bucket_name: str, last_run_at: str, processed_count: int) -> None:
    payload = json.dumps({"last_run_at": last_run_at, "processed_count": processed_count})
    storage.Client().bucket(bucket_name).blob(STATE_BLOB).upload_from_string(payload, content_type="application/json")


def _count_new_blobs(bucket_name: str, since: datetime | None) -> int:
    """Count prediction PNGs uploaded after `since` (or all if since is None)."""
    blobs = [
        b for b in storage.Client().bucket(bucket_name).list_blobs(prefix="predictions/") if b.name.endswith(".png")
    ]
    if since is None:
        return len(blobs)
    return sum(1 for b in blobs if b.time_created.replace(tzinfo=timezone.utc) > since)


# ---------------------------------------------------------------------------
# CLIP + embedding helpers
# ---------------------------------------------------------------------------


def _load_clip(device: str) -> tuple[CLIPModel, CLIPProcessor]:
    print(f"Loading CLIP ({CLIP_MODEL_ID})...")
    model: CLIPModel = CLIPModel.from_pretrained(CLIP_MODEL_ID)  # type: ignore[assignment]
    model = model.to(device)  # type: ignore[arg-type]
    model.eval()
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
    return model, processor


def _embed(images: list[Image.Image], model: CLIPModel, processor: CLIPProcessor, device: str) -> np.ndarray:
    all_emb = []
    for i in range(0, len(images), 32):
        batch = images[i : i + 32]
        inputs = processor(images=batch, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            vision_out = model.vision_model(pixel_values=inputs["pixel_values"])
            features = model.visual_projection(vision_out.pooler_output)
        all_emb.append(features.cpu().numpy())
    return np.vstack(all_emb)


def _download_reference(bucket_name: str) -> tuple[np.ndarray, np.ndarray]:
    print("Downloading reference embeddings from GCS...")
    client = storage.Client()
    with tempfile.NamedTemporaryFile(suffix=".npz") as f:
        client.bucket(bucket_name).blob("reference/embeddings.npz").download_to_filename(f.name)
        data = np.load(f.name)
        return data["embeddings"], data["targets"]


def _download_prediction_images(bucket_name: str, n: int) -> tuple[list[Image.Image], list[int]]:
    client = storage.Client()
    blobs = sorted(
        [b for b in client.bucket(bucket_name).list_blobs(prefix="predictions/") if b.name.endswith(".png")],
        key=lambda b: b.name,
    )[-n:]
    if not blobs:
        raise HTTPException(status_code=404, detail="No prediction images found in bucket.")
    images, targets = [], []
    with tempfile.TemporaryDirectory() as tmpdir:
        for blob in blobs:
            local = Path(tmpdir) / Path(blob.name).name
            blob.download_to_filename(local)
            images.append(Image.open(local).convert("RGB"))
            class_name = local.stem.split("_", 1)[-1]
            targets.append(CLASS_NAMES.index(class_name) if class_name in CLASS_NAMES else -1)
    return images, targets


def _run_analysis(n: int) -> str:
    """Run CLIP → PCA → Evidently and return HTML string."""
    cur_images, cur_targets = _download_prediction_images(MONITORING_BUCKET, n)
    cur_embeddings = _embed(cur_images, app.state.model, app.state.processor, app.state.device)
    cur_reduced = app.state.pca.transform(cur_embeddings)
    cur_df = pd.DataFrame(cur_reduced, columns=app.state.col_names)
    cur_df["target"] = cur_targets

    evidently_report = Report(metrics=[DataDriftPreset(), TargetDriftPreset()])
    evidently_report.run(reference_data=app.state.ref_df, current_data=cur_df)

    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
        evidently_report.save_html(f.name)
        return Path(f.name).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load CLIP, download the reference embeddings, and fit PCA on startup.

    Args:
        app: The FastAPI application; the models and reference data are
            stored on ``app.state``.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, processor = _load_clip(device)
    ref_embeddings, ref_targets = _download_reference(MONITORING_BUCKET)

    print(f"Fitting PCA ({N_COMPONENTS} components)...")
    pca = PCA(n_components=N_COMPONENTS, random_state=42)
    ref_reduced = pca.fit_transform(ref_embeddings)

    col_names = [f"pc{i}" for i in range(N_COMPONENTS)]
    ref_df = pd.DataFrame(ref_reduced, columns=col_names)
    ref_df["target"] = ref_targets.tolist()

    app.state.model = model
    app.state.processor = processor
    app.state.device = device
    app.state.pca = pca
    app.state.ref_df = ref_df
    app.state.col_names = col_names
    print("Monitoring API ready.")
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    """Report service liveness.

    Returns:
        ``{"status": "ok"}`` while the service is running.
    """
    return {"status": "ok"}


@app.get("/report", response_class=HTMLResponse)
def report(n: int = 500) -> HTMLResponse:
    """Run the full drift analysis unconditionally.

    Args:
        n: Number of most recent predictions to analyse.

    Returns:
        The Evidently drift report as HTML.
    """
    return HTMLResponse(content=_run_analysis(n))


@app.get("/check")
def check(n: int = 500) -> JSONResponse:
    """Run the drift analysis if enough new data has arrived (for Cloud Scheduler).

    Runs only when ``MIN_SAMPLES`` new predictions have arrived or
    ``MAX_STALENESS_HOURS`` have passed since the last full run; generated
    reports are archived to GCS.

    Args:
        n: Number of most recent predictions to analyse.

    Returns:
        Whether the analysis ran or was skipped, and why.
    """
    now = datetime.now(timezone.utc)
    state = _read_state(MONITORING_BUCKET)

    last_run_at = (
        datetime.fromisoformat(state["last_run_at"]).replace(tzinfo=timezone.utc) if state["last_run_at"] else None
    )

    new_count = _count_new_blobs(MONITORING_BUCKET, since=last_run_at)
    hours_since = (now - last_run_at).total_seconds() / 3600 if last_run_at else float("inf")

    stale = hours_since >= MAX_STALENESS_HOURS
    enough_samples = new_count >= MIN_SAMPLES

    if not enough_samples and not stale:
        msg = (
            f"Skipped: {new_count} new predictions (need {MIN_SAMPLES}), "
            f"{hours_since:.1f}h since last run (max {MAX_STALENESS_HOURS}h)."
        )
        print(msg)
        return JSONResponse({"status": "skipped", "reason": msg})

    reason = "enough_samples" if enough_samples else "max_staleness"
    print(f"Running analysis: {reason} — {new_count} new samples, {hours_since:.1f}h since last run.")

    html = _run_analysis(n)

    _write_state(MONITORING_BUCKET, now.isoformat(), state["processed_count"] + new_count)

    # Save report to GCS for history
    report_blob = f"reports/{now.strftime('%Y%m%dT%H%M%S')}_drift_report.html"
    storage.Client().bucket(MONITORING_BUCKET).blob(report_blob).upload_from_string(html, content_type="text/html")
    print(f"Report saved to gs://{MONITORING_BUCKET}/{report_blob}")

    return JSONResponse(
        {
            "status": "ran",
            "reason": reason,
            "new_samples": new_count,
            "report": f"gs://{MONITORING_BUCKET}/{report_blob}",
        }
    )
