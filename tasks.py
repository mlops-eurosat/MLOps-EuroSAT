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


# Data commands
@task
def preprocess_data(ctx: Context) -> None:
    """Regenerate the processed data through DVC and push it to the remote.

    Runs the preprocess stage so dvc.lock stays in sync with the data;
    commit the lock change afterwards.
    """
    ctx.run("dvc repro preprocess", echo=True, pty=not WINDOWS)
    ctx.run("dvc push", echo=True, pty=not WINDOWS)


@task
def dataset_statistics(ctx: Context) -> None:
    """Print dataset statistics and save distribution figures to reports/figures/."""
    ctx.run(f"python -m {PROJECT_NAME}.dataset_statistics", echo=True, pty=not WINDOWS)


# Training commands
@task
def train(ctx: Context) -> None:
    """Train model."""
    ctx.run(f"python src/{PROJECT_NAME}/train.py", echo=True, pty=not WINDOWS)


@task
def evaluate(ctx: Context, checkpoint: str) -> None:
    """Evaluate a trained checkpoint on the test set."""
    ctx.run(f"python src/{PROJECT_NAME}/evaluate.py {checkpoint}", echo=True, pty=not WINDOWS)


@task
def eval_report(ctx: Context, model: str = "", dummy: bool = False) -> None:
    """Render the evaluation report for the staging model to a local HTML file.

    --model uses a local ONNX file instead of downloading from the registry,
    --dummy fills the report with random data for a quick styling check.
    """
    cmd = f"python src/{PROJECT_NAME}/visualize.py"
    if model:
        cmd += f" --model {model}"
    if dummy:
        cmd += " --dummy"
    ctx.run(cmd, echo=True, pty=not WINDOWS)


@task
def profile(ctx: Context, steps: int = 50, num_workers: int = -1, torch_profiler: bool = True) -> None:
    """Profile the training pipeline (prints results to the terminal).

    --num-workers overrides the configured DataLoader workers, --no-torch-profiler
    skips the torch.profiler pass. e.g. invoke profile --steps 100 --num-workers 0
    """
    cmd = f"python src/{PROJECT_NAME}/profiling.py +profiling.steps={steps}"
    if num_workers >= 0:
        cmd += f" training.num_workers={num_workers}"
    if not torch_profiler:
        cmd += " +profiling.torch=false"
    ctx.run(cmd, echo=True, pty=not WINDOWS)


@task
def quantize(ctx: Context, onnx: str = "", checkpoint: str = "", batch_size: int = 1, runs: int = 200) -> None:
    """Quantize the ONNX model to int8 and benchmark fp32 vs int8 (size/latency/accuracy).

    Exports a fresh model when --onnx is omitted. e.g.
        invoke quantize
        invoke quantize --onnx models/model.onnx --batch-size 32
    """
    cmd = f"python src/{PROJECT_NAME}/quantize.py --batch-size {batch_size} --runs {runs}"
    if onnx:
        cmd += f" --onnx {onnx}"
    if checkpoint:
        cmd += f" --checkpoint {checkpoint}"
    ctx.run(cmd, echo=True, pty=not WINDOWS)


# Cloud training commands
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


# Pipeline & registry commands
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
    # The dvc.lock baked into the image is stale once preprocess reruns, so preprocess hands
    # the fresh lock to the later steps through GCS; they then pull exactly this run's data.
    lock_uri = f"{pl.PIPELINE_ROOT}/dvc-locks/{datetime.now():%Y%m%d-%H%M%S}/dvc.lock"

    def _lock_cmd(action: str) -> str:
        return (
            'python -c "from google.cloud import storage; '
            f"storage.Blob.from_uri('{lock_uri}', client=storage.Client()).{action}('dvc.lock')\""
        )

    preprocess_cmd = (
        "dvc pull data/raw.dvc -j 16 && dvc repro -f preprocess && dvc push -j 16 && "
        f"{_lock_cmd('upload_from_filename')}"
    )
    train_cmd = (
        f"{_lock_cmd('download_to_filename')} && dvc pull data/processed -j 16 && "
        f"python -u src/mlops_eurosat/train.py{overrides}"
    )

    def _worker_pool_specs(cmd: str) -> list:
        return [
            {
                "machine_spec": {"machine_type": machine_type},
                "replica_count": 1,
                "container_spec": {
                    "image_uri": pl.TRAIN_IMAGE,
                    "command": ["bash", "-c"],
                    "args": [cmd],
                },
            }
        ]

    package = pl.compile_pipeline()

    aiplatform.init(project=pl.PROJECT_ID, location=pl.REGION)
    job = aiplatform.PipelineJob(
        display_name="eurosat-training-pipeline",
        template_path=package,
        pipeline_root=pl.PIPELINE_ROOT,
        parameter_values={
            "preprocess_specs": _worker_pool_specs(preprocess_cmd),
            "worker_pool_specs": _worker_pool_specs(train_cmd),
            "lock_uri": lock_uri,
        },
    )
    job.run(sync=False)
    print("Submitted pipeline job to Vertex AI.")


@task
def registry_status(ctx: Context) -> None:
    """Show the staging and production versions in the model registry."""
    from mlops_eurosat import model_registry as mr

    for alias in (mr.STAGING_ALIAS, mr.PRODUCTION_ALIAS):
        try:
            model = mr._versioned_model(alias)
        except Exception:
            print(f"{alias:<11} -")
            continue
        acc = mr._decode_metric(model.labels.get(mr.METRIC_LABEL))
        acc_str = f"val_acc={acc:.4f}" if acc is not None else "no metric"
        print(f"{alias:<11} version {model.version_id} ({acc_str})")


@task
def promote(ctx: Context, max_latency_s: float = 5.0) -> None:
    """Run the gate -> promote -> deploy chain by hand.

    Same logic the registry trigger runs; useful when the Eventarc event
    did not arrive or a promotion should be retried.
    """
    from mlops_eurosat import model_registry

    status = model_registry.gate_promote_deploy(max_latency_s=max_latency_s)
    print(f"Result: {status}")


# Monitoring commands
@task
def drift_reference(ctx: Context, train_pt: str = "data/processed/train.pt") -> None:
    """Recompute the CLIP reference embeddings and upload them to the monitoring bucket.

    Rerun after every data change so drift reports compare against the current
    training distribution.
    """
    from mlops_eurosat.data_drift import save_reference_embeddings

    save_reference_embeddings(train_pt=Path(train_pt))


@task
def data_drift(ctx: Context, n_predictions: int = 500, output: str = "reports/drift_report.html") -> None:
    """Generate the offline drift report from the logged predictions.

    e.g. invoke data-drift --n-predictions 200
    """
    cmd = f"python src/{PROJECT_NAME}/data_drift.py --n-predictions {n_predictions} --output {output}"
    ctx.run(cmd, echo=True, pty=not WINDOWS)


# Quality commands
@task
def test(ctx: Context) -> None:
    """Run tests (coverage comes from the pytest-cov addopts in pyproject.toml)."""
    ctx.run("pytest tests/", echo=True, pty=not WINDOWS)


@task
def lint(ctx: Context) -> None:
    """Run ruff and mypy the same way CI does, fixing what ruff can."""
    ctx.run("ruff check . --fix", echo=True, pty=not WINDOWS)
    ctx.run("ruff format .", echo=True, pty=not WINDOWS)
    ctx.run("mypy .", echo=True, pty=not WINDOWS)


# Serving commands
@task
def serve_api(ctx: Context, port: int = 8000) -> None:
    """Serve the prediction API locally with auto-reload.

    The model is fetched from AIP_STORAGE_URI on startup; without GCP
    credentials /predict answers 503 but the API still runs.
    """
    ctx.run(f"uvicorn {PROJECT_NAME}.api:app --reload --port {port}", echo=True, pty=not WINDOWS)


@task
def frontend(ctx: Context) -> None:
    """Run the Streamlit frontend (live map + upload tab).

    The map needs Sentinel Hub credentials; if SH_CLIENT_ID or
    SH_CLIENT_SECRET is not set, it is fetched from Secret Manager (the same
    secrets the Cloud Run deploy uses).
    """
    env = {}
    for var, secret in (("SH_CLIENT_ID", "sh-client-id"), ("SH_CLIENT_SECRET", "sh-client-secret")):
        if not os.environ.get(var):
            fetched = ctx.run(f"gcloud secrets versions access latest --secret={secret}", hide=True)
            env[var] = fetched.stdout.strip()
    ctx.run(f"streamlit run src/{PROJECT_NAME}/frontend_map.py", echo=True, pty=not WINDOWS, env=env)


# Build & deploy commands
@task
def docker_build(ctx: Context, progress: str = "plain") -> None:
    """Build all docker images."""
    for image in ("train", "api", "frontend", "monitoring", "trigger"):
        ctx.run(
            f"docker build -t {image}:latest . -f dockerfiles/{image}.dockerfile --progress={progress}",
            echo=True,
            pty=not WINDOWS,
        )


@task
def cloud_build(ctx: Context) -> None:
    """Build, push and deploy all images through Cloud Build (same as the merge trigger)."""
    ctx.run("gcloud builds submit --config cloudbuild.yaml .", echo=True, pty=not WINDOWS)


# Documentation commands
@task(dev_requirements)
def build_docs(ctx: Context) -> None:
    """Build documentation."""
    ctx.run("mkdocs build --config-file docs/mkdocs.yaml --site-dir build", echo=True, pty=not WINDOWS)


@task(dev_requirements)
def serve_docs(ctx: Context) -> None:
    """Serve documentation."""
    ctx.run("mkdocs serve --config-file docs/mkdocs.yaml", echo=True, pty=not WINDOWS)
