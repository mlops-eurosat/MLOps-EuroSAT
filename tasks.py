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
    """Submit a Vertex AI custom training job (W&B key from .env)."""
    key = ""
    for line in Path(".env").read_text().splitlines():
        if line.strip().startswith("WANDB_API_KEY="):
            key = line.split("=", 1)[1].strip()
    config = Path("vertex_config_cpu.yaml").read_text()
    filled = Path(tempfile.gettempdir()) / "vertex_filled.yaml"
    filled.write_text(config.replace("REPLACE_WITH_YOUR_KEY", key))
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
    """Submit a Vertex AI job that runs a W&B sweep agent (W&B key in .env)

    Create the sweep first with `invoke sweep-create` or
    `wandb sweep configs/sweep.yaml`, then pass the printed id. `count` is the
    number of runs the agent executes and is freely chosen, e.g.
    `invoke sweep-cloud --sweep-id mlops-eurosat/mlops-eurosat/abc123 --count 5`.
    """
    key = ""
    for line in Path(".env").read_text().splitlines():
        if line.strip().startswith("WANDB_API_KEY="):
            key = line.split("=", 1)[1].strip()
    config = Path("vertex_config_sweep.yaml").read_text()
    config = config.replace("REPLACE_WITH_YOUR_KEY", key)
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
def sweep_best(ctx: Context, sweep_id: str, promote: bool = False) -> None:
    """Fetch a sweep's best run, save its hyperparameters, and optionally promote it.

    Promotion uses a champion/challenger check: the model only becomes the new
    `best` if its val_acc beats the current best in the registry.
    e.g.
        invoke sweep-best --sweep-id mlops-eurosat/mlops-eurosat/abc123
        invoke sweep-best --sweep-id mlops-eurosat/mlops-eurosat/abc123 --promote
    """
    import wandb
    import yaml

    from mlops_eurosat import registry

    api = wandb.Api()
    sweep = api.sweep(sweep_id)
    best = sweep.best_run()
    entity, project = sweep.entity, sweep.project

    val_acc = best.summary.get(registry.METRIC_KEY)
    t = best.config.get("training", {})
    result = {
        "sweep": sweep_id,
        "run_name": best.name,
        "run_id": best.id,
        registry.METRIC_KEY: val_acc,
        "training": {"lr": t.get("lr"), "batch_size": t.get("batch_size"), "max_epochs": t.get("max_epochs")},
    }
    out = Path("outputs/best_hyperparameters.yaml")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(result, sort_keys=False))
    acc_str = f"{val_acc:.4f}" if isinstance(val_acc, (int, float)) else "unknown"
    print(f"Best run in sweep: {best.name} ({registry.METRIC_KEY}={acc_str})")
    print(f"Saved hyperparameters -> {out}")

    if not promote:
        print("Run again with --promote to promote this model to 'best'.")
        return

    if not isinstance(val_acc, (int, float)):
        print(f"Cannot promote: best run has no numeric '{registry.METRIC_KEY}'.")
        return

    promoted = registry.promote_if_better(entity, project, best.id)
    if promoted:
        print("This model is now the registry 'best'.")
    else:
        print("Existing 'best' model retained (candidate was not better).")


@task
def model_best(ctx: Context) -> None:
    """Show which model is currently registered as 'best' and its metric."""
    from mlops_eurosat import registry

    version, metric = registry.get_current_best()
    if version is None:
        print("No 'best' model registered yet.")
        return
    metric_str = f"{metric:.4f}" if isinstance(metric, (int, float)) else "unknown"
    print(f"Current best: {registry.MODEL_NAME}:{version}  ({registry.METRIC_KEY}={metric_str})")


@task
def model_download(ctx: Context, root: str = "models/best") -> None:
    """Download the current 'best' model artifact for local use."""
    from mlops_eurosat import registry

    registry.download_best(root=root)


@task
def promote_run(ctx: Context, run_id: str, entity: str = "mlops-eurosat", project: str = "mlops-eurosat") -> None:
    """Conditionally promote a single run's model to the registry 'best'.

    Use this when you trained one model (not a sweep) and want to register it.
    The model only becomes 'best' if its val_acc beats the current best.

    e.g. invoke promote-run --run-id abc123de
    """
    import wandb

    from mlops_eurosat import registry

    run = wandb.Api().run(f"{entity}/{project}/{run_id}")
    val_acc = run.summary.get(registry.METRIC_KEY)

    if not isinstance(val_acc, (int, float)):
        print(f"Cannot promote: run '{run_id}' has no numeric '{registry.METRIC_KEY}' in its summary.")
        return

    print(f"Run {run.name} ({run_id}): {registry.METRIC_KEY}={val_acc:.4f}")
    promoted = registry.promote_if_better(entity, project, run_id)
    if promoted:
        print("This model is now the registry 'best'")
    else:
        print("Existing 'best' model retained (candidate was not better)")


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
