"""Vertex AI Pipeline

train (Vertex CustomJob) -> registers the new model as ``staging``.

The training step runs our existing ``train`` container as a Vertex CustomJob;
``train.py`` uploads the best checkpoint to GCS and registers it in the Vertex
Model Registry under the ``staging`` alias. The downstream gate -> promote ->
deploy is not part of this pipeline: it is owned by the registry-change trigger
(``registry_trigger.py``), which fires whenever a model version is uploaded.

Compile + submit via ``invoke pipeline-run`` .
"""

from kfp import compiler, dsl

from mlops_eurosat import vertex_registry as vr

PROJECT_ID = vr.PROJECT_ID
REGION = vr.REGION
TRAIN_IMAGE = f"{REGION}-docker.pkg.dev/{PROJECT_ID}/mlops-eurosat/train:latest"
PIPELINE_ROOT = "gs://eurosat_models/pipeline-root"


@dsl.pipeline(name="eurosat-training-pipeline", pipeline_root=PIPELINE_ROOT)
def eurosat_pipeline(worker_pool_specs: list) -> None:
    from google_cloud_pipeline_components.v1.custom_job import CustomTrainingJobOp

    train_task = CustomTrainingJobOp(
        project=PROJECT_ID,
        location=REGION,
        display_name="eurosat-pipeline-train",
        worker_pool_specs=worker_pool_specs,
    )
    train_task.set_caching_options(False)


def compile_pipeline(path: str = "eurosat_pipeline.json") -> str:
    """Compile the pipeline to a JSON spec that Vertex can run."""
    compiler.Compiler().compile(pipeline_func=eurosat_pipeline, package_path=path)
    print(f"Compiled pipeline -> {path}")
    return path


if __name__ == "__main__":
    compile_pipeline()
