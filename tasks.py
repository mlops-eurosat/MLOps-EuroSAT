import os
import tempfile
from datetime import datetime
from pathlib import Path

from invoke import Context, task

WINDOWS = os.name == "nt"
PROJECT_NAME = "mlops_eurosat"
PYTHON_VERSION = "3.12"


# Setup commands
@task
def create_environment(ctx: Context) -> None:
    """Create a new conda environment for project."""
    ctx.run(
        f"conda create --name {PROJECT_NAME} python={PYTHON_VERSION} pip --no-default-packages --yes",
        echo=True,
        pty=not WINDOWS,
    )


@task
def requirements(ctx: Context) -> None:
    """Install project requirements."""
    ctx.run("pip install -U pip setuptools wheel", echo=True, pty=not WINDOWS)
    ctx.run("pip install -r requirements.txt", echo=True, pty=not WINDOWS)
    ctx.run("pip install -e .", echo=True, pty=not WINDOWS)


@task(requirements)
def dev_requirements(ctx: Context) -> None:
    """Install development requirements."""
    ctx.run('pip install -e .["dev"]', echo=True, pty=not WINDOWS)


# Project commands
@task
def preprocess_data(ctx: Context) -> None:
    """Preprocess data."""
    ctx.run(f"python src/{PROJECT_NAME}/data.py data/raw data/processed", echo=True, pty=not WINDOWS)


@task
def train(ctx: Context) -> None:
    """Train model."""
    ctx.run(f"python src/{PROJECT_NAME}/train.py", echo=True, pty=not WINDOWS)


@task
def train_cloud(ctx: Context) -> None:
    """Submit a Vertex AI custom training job"""
    config = Path("vertex_config_cpu.yaml").read_text()
    filled = Path(tempfile.gettempdir()) / "vertex_filled.yaml"
    filled.write_text(config)
    ctx.run(
        f"gcloud ai custom-jobs create --region=europe-west3 --display-name=eurosat-test-run --config={filled}",
        echo=True,
        pty=not WINDOWS,
    )


@task
def sweep_create(ctx: Context) -> None:
    """Create a W&B sweep from configs/sweep.yaml; prints the id."""
    import wandb
    import yaml

    config = yaml.safe_load(Path("configs/sweep.yaml").read_text())
    config["name"] = f"eurosat-{datetime.now():%Y%m%d-%H%M}"
    sweep_id = wandb.sweep(config, project=config["project"], entity=config["entity"])
    full_id = f"{config['entity']}/{config['project']}/{sweep_id}"
    print(f"\nSweep created: {config['name']}")
    print(f"full id: {full_id}")
    print(f"launch:  invoke sweep-cloud --sweep-id {full_id} --count N")


@task
def sweep_cloud(ctx: Context, sweep_id: str, count: int = 10) -> None:
    """Submit a Vertex AI job that runs a W&B sweep agent

    Create the sweep first with `invoke sweep-create` or
    `wandb sweep configs/sweep.yaml`, then pass the printed id. `count` is the
    number of runs the agent executes and is freely chosen, e.g.
    `invoke sweep-cloud --sweep-id mlops-eurosat/mlops-eurosat/abc123 --count 5`.
    """
    config = Path("vertex_config_sweep.yaml").read_text()
    config = config.replace("COUNT_PLACEHOLDER", str(count))
    config = config.replace("SWEEP_ID_PLACEHOLDER", sweep_id)
    filled = Path(tempfile.gettempdir()) / "vertex_sweep_filled.yaml"
    filled.write_text(config)
    display_name = f"eurosat-sweep-{datetime.now():%Y%m%d-%H%M}-{count}runs"
    ctx.run(
        f"gcloud ai custom-jobs create --region=europe-west3 --display-name={display_name} --config={filled}",
        echo=True,
        pty=not WINDOWS,
    )


@task
def sweep_best(ctx: Context, sweep_id: str) -> None:
    """Fetch a sweep's best run and save its hyperparameters to outputs/.

    HPO is tracking-only. Retrain the printed config once with
    `register_model=true` to register the model in the Vertex Model Registry.
    e.g. invoke sweep-best --sweep-id mlops-eurosat/mlops-eurosat/abc123
    """
    import wandb
    import yaml

    metric_key = "val_acc"
    api = wandb.Api()
    sweep = api.sweep(sweep_id)
    best = sweep.best_run()

    val_acc = best.summary.get(metric_key)
    t = best.config.get("training", {})
    result = {
        "sweep": sweep_id,
        "run_name": best.name,
        "run_id": best.id,
        metric_key: val_acc,
        "training": {"lr": t.get("lr"), "batch_size": t.get("batch_size"), "max_epochs": t.get("max_epochs")},
    }
    out = Path("outputs/best_hyperparameters.yaml")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(result, sort_keys=False))
    acc_str = f"{val_acc:.4f}" if isinstance(val_acc, (int, float)) else "unknown"
    print(f"Best run in sweep: {best.name} ({metric_key}={acc_str})")
    print(f"Saved hyperparameters -> {out}")
    print("Register this config in Vertex by retraining once:")
    print(
        f"  python src/mlops_eurosat/train.py training.lr={t.get('lr')} "
        f"training.batch_size={t.get('batch_size')} training.register_model=true"
    )


@task
def pipeline_run(
    ctx: Context,
    machine_type: str = "n1-standard-32",
    smoke: bool = False,
    epochs: int = 0,
    limit_batches: str = "",
) -> None:
    """Compile and submit the Vertex AI training pipeline (trains + registers staging).

    The downstream gate -> promote -> deploy is handled by the registry trigger.
    The training container fetches WANDB_API_KEY from Secret Manager .

    For a fast end-to-end smoke test use --smoke (2 epochs, 10% of batches), or
    override individually with --epochs / --limit-batches.
    e.g.
        invoke pipeline-run                 # full training run
        invoke pipeline-run --smoke         # quick check that the chain works
        invoke pipeline-run --epochs 5 --limit-batches 0.2
    """
    from google.cloud import aiplatform

    from mlops_eurosat import pipeline as pl

    if smoke:
        epochs = epochs or 2
        limit_batches = limit_batches or "0.1"
    overrides = ""
    if epochs:
        overrides += f" training.max_epochs={epochs}"
    if limit_batches:
        overrides += f" training.limit_train_batches={limit_batches}"
    train_cmd = f"dvc pull && python -u src/mlops_eurosat/train.py{overrides}"

    package = pl.compile_pipeline()
    worker_pool_specs = [
        {
            "machine_spec": {"machine_type": machine_type},
            "replica_count": 1,
            "container_spec": {
                "image_uri": pl.TRAIN_IMAGE,
                "command": ["bash", "-c"],
                "args": [train_cmd],
            },
        }
    ]

    aiplatform.init(project=pl.PROJECT_ID, location=pl.REGION)
    job = aiplatform.PipelineJob(
        display_name="eurosat-training-pipeline",
        template_path=package,
        pipeline_root=pl.PIPELINE_ROOT,
        parameter_values={"worker_pool_specs": worker_pool_specs},
    )
    job.run(sync=False)
    print("Submitted pipeline job to Vertex AI.")


@task
def test(ctx: Context) -> None:
    """Run tests."""
    ctx.run("coverage run -m pytest tests/", echo=True, pty=not WINDOWS)
    ctx.run("coverage report -m -i", echo=True, pty=not WINDOWS)


@task
def docker_build(ctx: Context, progress: str = "plain") -> None:
    """Build docker images."""
    ctx.run(
        f"docker build -t train:latest . -f dockerfiles/train.dockerfile --progress={progress}",
        echo=True,
        pty=not WINDOWS,
    )
    ctx.run(
        f"docker build -t api:latest . -f dockerfiles/api.dockerfile --progress={progress}", echo=True, pty=not WINDOWS
    )


# Documentation commands
@task(dev_requirements)
def build_docs(ctx: Context) -> None:
    """Build documentation."""
    ctx.run("mkdocs build --config-file docs/mkdocs.yaml --site-dir build", echo=True, pty=not WINDOWS)


@task(dev_requirements)
def serve_docs(ctx: Context) -> None:
    """Serve documentation."""
    ctx.run("mkdocs serve --config-file docs/mkdocs.yaml", echo=True, pty=not WINDOWS)
