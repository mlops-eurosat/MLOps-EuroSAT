"""Result visualizations for the EuroSAT classifier.

Used by the ``evaluate`` pipeline step to turn test-set predictions into a
per-class metrics table and a grid of misclassified examples, embedded into a
single HTML artifact. The confusion matrix itself is logged separately as a
native Vertex ``ClassificationMetrics`` artifact.
"""

from __future__ import annotations

import base64
import io
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from sklearn.metrics import precision_recall_fscore_support  # noqa: E402


def misclassified_grid_figure(
    images: torch.Tensor,
    targets: list[int],
    preds: list[int],
    classes: list[str],
    mean: torch.Tensor,
    std: torch.Tensor,
    n: int = 16,
) -> Figure:
    """Grid of misclassified test images with predicted vs. true labels."""
    targets_t = torch.tensor(targets)
    preds_t = torch.tensor(preds)
    wrong = (targets_t != preds_t).nonzero(as_tuple=True)[0][:n]

    mean = mean.view(1, 3, 1, 1)
    std = std.view(1, 3, 1, 1)

    cols = 4
    rows = (len(wrong) + cols - 1) // cols or 1
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3.4 * rows), squeeze=False)
    for ax, idx in zip(axes.flat, wrong):
        img = ((images[idx].unsqueeze(0) * std + mean).clamp(0, 1))[0]
        ax.imshow(img.permute(1, 2, 0).cpu())
        ax.set_xticks([])
        ax.set_yticks([])
        # pred + true both above the image, colour-coded.
        ax.text(
            0.5,
            1.14,
            f"pred: {classes[preds_t[idx]]}",
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=8,
            color="#d62728",
        )
        ax.text(
            0.5,
            1.02,
            f"true: {classes[targets_t[idx]]}",
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=8,
            color="#2ca02c",
        )
    for ax in axes.flat[len(wrong) :]:
        ax.axis("off")
    fig.subplots_adjust(hspace=0.5, wspace=0.1)
    return fig


def per_class_table(targets: list[int], preds: list[int], classes: list[str]) -> str:
    """HTML table of precision / recall / F1 and sample count (support) per class."""
    precision, recall, f1, support = precision_recall_fscore_support(
        targets, preds, labels=list(range(len(classes))), zero_division=0
    )
    header = "<tr><th>class</th><th>precision</th><th>recall</th><th>f1</th><th>support</th></tr>"
    body = "".join(
        f"<tr><td>{name}</td><td class='val'>{precision[i]:.3f}</td><td class='val'>{recall[i]:.3f}</td>"
        f"<td class='val'>{f1[i]:.3f}</td><td class='val'>{int(support[i])}</td></tr>"
        for i, name in enumerate(classes)
    )
    return f"<table class='metrics'>{header}{body}</table>"


_CSS = """
body { font-family: -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
       margin: 0; padding: 2rem; color: #1a1a1a; background: #f6f7f9; }
.wrap { max-width: 1100px; margin: 0 auto; }
h1 { font-size: 1.6rem; margin: 0 0 1.5rem; }
h2 { font-size: 1.1rem; margin: 0 0 .75rem; color: #333; }
.card { background: #fff; border: 1px solid #e6e6e6; border-radius: 12px;
        padding: 1.25rem; margin-bottom: 1.5rem; box-shadow: 0 1px 3px rgba(0,0,0,.06); }
.card img { width: 100%; height: auto; display: block; }
table.metrics { border-collapse: collapse; }
table.metrics td, table.metrics th { padding: .45rem 1.1rem; text-align: left; border-bottom: 1px solid #eee; }
table.metrics th { color: #666; font-weight: 600; }
.val { font-variant-numeric: tabular-nums; font-weight: 600; }
"""


def figures_to_html(
    figures: dict[str, Figure],
    title: str = "EuroSAT evaluation report",
    meta: dict[str, str] | None = None,
    metrics: dict[str, float] | None = None,
    tables: dict[str, str] | None = None,
) -> str:
    """Render model metadata, metrics, tables and figures into a styled HTML document."""
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        f"<style>{_CSS}</style></head><body><div class='wrap'>",
        f"<h1>{title}</h1>",
    ]

    if meta:
        rows = "".join(f"<tr><th>{name}</th><td>{value}</td></tr>" for name, value in meta.items())
        parts.append(f"<div class='card'><h2>Model</h2><table class='metrics'>{rows}</table></div>")

    if metrics:
        rows = "".join(f"<tr><th>{name}</th><td class='val'>{value:.4f}</td></tr>" for name, value in metrics.items())
        parts.append(f"<div class='card'><h2>Summary metrics</h2><table class='metrics'>{rows}</table></div>")

    for name, table_html in (tables or {}).items():
        parts.append(f"<div class='card'><h2>{name}</h2>{table_html}</div>")

    for name, fig in figures.items():
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", bbox_inches="tight", dpi=110)
        plt.close(fig)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        parts.append(f"<div class='card'><h2>{name}</h2><img src='data:image/png;base64,{encoded}'/></div>")

    parts.append("</div></body></html>")
    return "\n".join(parts)


WANDB_PROJECT_URL = "https://wandb.ai/mlops-eurosat/mlops-eurosat"


def traceability_meta(artifact_uri: str, trained: datetime) -> dict[str, str]:
    """Model-header metadata: GCS artifact, W&B run link, and training time (local).

    The training run id is the last path segment of the artifact URI
    (``gs://.../checkpoints/<run_id>``). ``trained`` is the model version's
    registration time; ``.astimezone()`` shows it in local time (not UTC).
    """
    run_id = artifact_uri.rstrip("/").split("/")[-1]
    return {
        "artifact": artifact_uri,
        "W&B run": f"<a href='{WANDB_PROJECT_URL}/runs/{run_id}'>{run_id}</a>",
        "trained": trained.astimezone().strftime("%Y-%m-%d %H:%M"),
    }


def _download_staging_onnx(dest: str = "/tmp/model.onnx") -> tuple[str, str, datetime]:
    """Download the staging ``model.onnx`` from GCS; return (path, artifact_uri, trained)."""
    from google.cloud import aiplatform, storage  # type: ignore[attr-defined]

    from mlops_eurosat import model_registry as mr

    aiplatform.init(project=mr.PROJECT_ID, location=mr.REGION)
    base = aiplatform.Model.list(filter=f'display_name="{mr.MODEL_DISPLAY_NAME}"')[0]
    staging = aiplatform.Model(model_name=base.resource_name, version=mr.STAGING_ALIAS)
    uri = staging.uri
    if not Path(dest).exists():
        bucket, _, prefix = uri[len("gs://") :].partition("/")
        storage.Client().bucket(bucket).blob(f"{prefix.rstrip('/')}/model.onnx").download_to_filename(dest)
        print(f"downloaded {uri}/model.onnx -> {dest}")
    return dest, uri, staging.version_create_time


def main(
    model: str = "",
    data: str = "data/processed/test.pt",
    out: str = "reports/figures/eval_report.html",
    dummy: bool = False,
) -> None:
    """Render the evaluation report to a local HTML file for inspection.

    Default: download the staging ONNX model (or use ``--model``) and run inference
    on the processed test set. With ``--dummy`` use random data for a quick styling
    preview (no model/data/GCP needed).
    """
    scores: dict[str, float] | None = None
    figures: dict[str, Figure] = {}
    staging_info: tuple[str, datetime] | None = None

    if dummy:
        num_classes = 10
        classes = [f"class{i}" for i in range(num_classes)]
        targets = [i % num_classes for i in range(400)]
        preds = [(i * 7) % num_classes for i in range(400)]
        images = torch.rand(400, 3, 64, 64)
        mean = torch.tensor([0.5, 0.5, 0.5])
        std = torch.tensor([0.2, 0.2, 0.2])
    else:
        import numpy as np
        import onnxruntime as ort
        from sklearn.metrics import accuracy_score, f1_score

        payload = torch.load(data, weights_only=False)
        classes = payload["classes"]
        images = payload["images"]
        mean, std = payload["mean"], payload["std"]
        targets = payload["targets"].tolist()

        if model:
            model_path = model
        else:
            model_path, uri, trained = _download_staging_onnx()
            staging_info = (uri, trained)
        session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name
        arr = images.numpy().astype("float32")
        preds = []
        for start in range(0, len(arr), 256):
            logits = session.run(None, {input_name: arr[start : start + 256]})[0]
            preds.extend(np.asarray(logits).argmax(axis=1).tolist())

        scores = {
            "accuracy": float(accuracy_score(targets, preds)),
            "macro_f1": float(f1_score(targets, preds, average="macro")),
            "weighted_f1": float(f1_score(targets, preds, average="weighted")),
        }

    figures["Misclassified examples"] = misclassified_grid_figure(images, targets, preds, classes, mean, std)

    if staging_info:
        meta = traceability_meta(*staging_info)
    else:
        meta = {"model": model or "dummy data"}
    tables = {"Per-class metrics": per_class_table(targets, preds, classes)}

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(figures_to_html(figures, meta=meta, metrics=scores, tables=tables))
    print(f"wrote {out_path.resolve()}")


if __name__ == "__main__":
    import typer

    typer.run(main)
