"""Vertex AI Pipeline

preprocess -> train -> evaluate. train registers the new model as ``staging``.

The preprocess step regenerates the processed data via DVC; train runs after it.
The evaluate step scores the staging model on the test set and logs metrics.
The training step runs our existing ``train`` container as a Vertex CustomJob;
``train.py`` uploads the best checkpoint to GCS and registers it in the Vertex
Model Registry under the ``staging`` alias. The downstream gate -> promote ->
deploy is not part of this pipeline: it is owned by the registry-change trigger
(``registry_trigger.py``), which fires whenever a model version is uploaded.

Compile + submit via ``invoke pipeline-run`` .
"""

from kfp import compiler, dsl
from kfp.dsl import HTML, ClassificationMetrics, Metrics, Output

from mlops_eurosat import model_registry as mr

PROJECT_ID = mr.PROJECT_ID
REGION = mr.REGION
TRAIN_IMAGE = f"{REGION}-docker.pkg.dev/{PROJECT_ID}/mlops-eurosat/train:latest"
PIPELINE_ROOT = "gs://eurosat_models/pipeline-root"


@dsl.component(base_image=TRAIN_IMAGE)
def evaluate_model(
    metrics: Output[Metrics],
    classification_metrics: Output[ClassificationMetrics],
    plots: Output[HTML],
) -> None:
    """Score the freshly registered ``staging`` model on the test set, log metrics
    and result visualizations (confusion matrix, per-class metrics, misclassified
    examples)."""
    import subprocess

    import numpy as np
    import onnxruntime as ort
    import torch
    from google.cloud import aiplatform, storage  # type: ignore[attr-defined]
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

    from mlops_eurosat import model_registry as mr
    from mlops_eurosat import visualize

    # Test data + the freshly registered staging model (served as ONNX).
    subprocess.run(["dvc", "pull", "data/processed"], cwd="/app", check=True)

    aiplatform.init(project=mr.PROJECT_ID, location=mr.REGION)
    base = aiplatform.Model.list(filter=f'display_name="{mr.MODEL_DISPLAY_NAME}"')[0]
    staging = aiplatform.Model(model_name=base.resource_name, version=mr.STAGING_ALIAS)
    bucket, _, prefix = staging.uri[len("gs://") :].partition("/")
    storage.Client().bucket(bucket).blob(f"{prefix.rstrip('/')}/model.onnx").download_to_filename("/app/model.onnx")

    session = ort.InferenceSession("/app/model.onnx", providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    data = torch.load("/app/data/processed/test.pt", weights_only=False)
    classes = data["classes"]
    images = data["images"].numpy().astype("float32")
    targets = data["targets"].tolist()

    preds: list[int] = []
    for start in range(0, len(images), 256):
        logits = session.run(None, {input_name: images[start : start + 256]})[0]
        preds.extend(np.asarray(logits).argmax(axis=1).tolist())

    scores = {
        "accuracy": float(accuracy_score(targets, preds)),
        "macro_f1": float(f1_score(targets, preds, average="macro")),
        "weighted_f1": float(f1_score(targets, preds, average="weighted")),
    }
    for name, value in scores.items():
        metrics.log_metric(name, value)

    # Confusion matrix -> rendered natively by Vertex.
    matrix = confusion_matrix(targets, preds, labels=list(range(len(classes))))
    classification_metrics.log_confusion_matrix(classes, matrix.tolist())

    # Model header (artifact + W&B run link + training time)
    meta = visualize.traceability_meta(staging.uri, staging.version_create_time)
    tables = {"Per-class metrics": visualize.per_class_table(targets, preds, classes)}

    # Model header + summary metrics + per-class table + figures -> single HTML artifact.
    figures = {
        "Misclassified examples": visualize.misclassified_grid_figure(
            data["images"], targets, preds, classes, data["mean"], data["std"]
        ),
    }
    with open(plots.path, "w") as f:
        f.write(visualize.figures_to_html(figures, meta=meta, metrics=scores, tables=tables))


@dsl.pipeline(name="eurosat-training-pipeline", pipeline_root=PIPELINE_ROOT)
def eurosat_pipeline(preprocess_specs: list, worker_pool_specs: list) -> None:
    from google_cloud_pipeline_components.v1.custom_job import CustomTrainingJobOp

    preprocess_task = CustomTrainingJobOp(
        project=PROJECT_ID,
        location=REGION,
        display_name="eurosat-pipeline-preprocess",
        worker_pool_specs=preprocess_specs,
    )
    preprocess_task.set_caching_options(
        True
    )  # Set to False for final training runs, only True for faster iteration during development.
    preprocess_task.set_display_name("preprocess")

    train_task = CustomTrainingJobOp(
        project=PROJECT_ID,
        location=REGION,
        display_name="eurosat-pipeline-train",
        worker_pool_specs=worker_pool_specs,
    )
    train_task.set_caching_options(False)
    train_task.set_display_name("train")
    train_task.after(preprocess_task)

    evaluate_task = evaluate_model()
    evaluate_task.set_caching_options(False)
    evaluate_task.set_display_name("evaluate")
    evaluate_task.after(train_task)


def compile_pipeline(path: str = "eurosat_pipeline.json") -> str:
    """Compile the pipeline to a JSON spec that Vertex can run."""
    compiler.Compiler().compile(pipeline_func=eurosat_pipeline, package_path=path)
    print(f"Compiled pipeline -> {path}")
    return path


if __name__ == "__main__":
    compile_pipeline()
