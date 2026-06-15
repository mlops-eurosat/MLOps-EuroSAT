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
from kfp.dsl import Metrics, Output

from mlops_eurosat import model_registry as mr

PROJECT_ID = mr.PROJECT_ID
REGION = mr.REGION
TRAIN_IMAGE = f"{REGION}-docker.pkg.dev/{PROJECT_ID}/mlops-eurosat/train:latest"
PIPELINE_ROOT = "gs://eurosat_models/pipeline-root"


@dsl.component(base_image=TRAIN_IMAGE)
def evaluate_model(metrics: Output[Metrics]) -> None:
    """Score the freshly registered ``staging`` model on the test set and log metrics."""
    import subprocess

    import torch
    from google.cloud import aiplatform, storage  # type: ignore[attr-defined]
    from sklearn.metrics import accuracy_score, f1_score
    from torch.utils.data import DataLoader, TensorDataset

    from mlops_eurosat import model_registry as mr
    from mlops_eurosat.model import Model

    # Test data + the freshly registered staging checkpoint.
    subprocess.run(["dvc", "pull", "data/processed"], cwd="/app", check=True)

    aiplatform.init(project=mr.PROJECT_ID, location=mr.REGION)
    base = aiplatform.Model.list(filter=f'display_name="{mr.MODEL_DISPLAY_NAME}"')[0]
    staging = aiplatform.Model(model_name=base.resource_name, version=mr.STAGING_ALIAS)
    bucket, _, prefix = staging.uri[len("gs://") :].partition("/")
    storage.Client().bucket(bucket).blob(f"{prefix.rstrip('/')}/model.ckpt").download_to_filename("/app/model.ckpt")

    model = Model()
    ckpt = torch.load("/app/model.ckpt", map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    data = torch.load("/app/data/processed/test.pt", weights_only=False)
    loader = DataLoader(TensorDataset(data["images"], data["targets"]), batch_size=32)

    preds: list[int] = []
    targets: list[int] = []
    with torch.no_grad():
        for images, labels in loader:
            preds.extend(model(images).argmax(dim=1).tolist())
            targets.extend(labels.tolist())

    metrics.log_metric("accuracy", float(accuracy_score(targets, preds)))
    metrics.log_metric("macro_f1", float(f1_score(targets, preds, average="macro")))
    metrics.log_metric("weighted_f1", float(f1_score(targets, preds, average="weighted")))


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
