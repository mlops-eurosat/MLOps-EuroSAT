"""Result visualizations for the EuroSAT classifier.

Used by the ``evaluate`` pipeline step to turn test-set predictions into figures
(per-class metrics, misclassified examples) that are embedded into a single HTML
artifact. The confusion matrix itself is logged separately as a native Vertex
``ClassificationMetrics`` artifact.
"""

from __future__ import annotations

import base64
import io

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from sklearn.metrics import precision_recall_fscore_support  # noqa: E402


def per_class_metrics_figure(targets: list[int], preds: list[int], classes: list[str]) -> Figure:
    """Bar chart of precision / recall / F1 per class."""
    labels = list(range(len(classes)))
    precision, recall, f1, _ = precision_recall_fscore_support(targets, preds, labels=labels, zero_division=0)

    x = torch.arange(len(classes))
    width = 0.25
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - width, precision, width, label="precision")
    ax.bar(x, recall, width, label="recall")
    ax.bar(x + width, f1, width, label="f1")
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("score")
    ax.set_title("Per-class precision / recall / F1")
    ax.legend()
    fig.tight_layout()
    return fig


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
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows), squeeze=False)
    for ax, idx in zip(axes.flat, wrong):
        img = ((images[idx].unsqueeze(0) * std + mean).clamp(0, 1))[0]
        ax.imshow(img.permute(1, 2, 0).cpu())
        ax.set_title(f"pred: {classes[preds_t[idx]]}\ntrue: {classes[targets_t[idx]]}", fontsize=8)
    for ax in axes.flat:
        ax.axis("off")
    fig.suptitle(f"Misclassified test images (showing {len(wrong)})")
    fig.tight_layout()
    return fig


def figures_to_html(figures: dict[str, Figure]) -> str:
    """Embed matplotlib figures as base64 PNGs into a single HTML document."""
    parts = ["<html><body>"]
    for title, fig in figures.items():
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", bbox_inches="tight")
        plt.close(fig)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        parts.append(f"<h2>{title}</h2><img src='data:image/png;base64,{encoded}'/>")
    parts.append("</body></html>")
    return "\n".join(parts)
