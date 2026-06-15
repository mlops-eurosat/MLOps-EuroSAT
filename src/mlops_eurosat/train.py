import logging
import os
from datetime import datetime
from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
from google.cloud import storage  # type: ignore[attr-defined]
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, TensorDataset

from mlops_eurosat import vertex_registry
from mlops_eurosat.model import Model

log = logging.getLogger(__name__)

MODEL_BUCKET = "eurosat_models"
SECRET_PROJECT = "mlops-eurosat-496913"
WANDB_SECRET_NAME = "wandb-api-key"


def _ensure_wandb_key() -> None:
    """Ensure WANDB_API_KEY is set, fetching it from Secret Manager if needed"""

    if os.getenv("WANDB_API_KEY"):
        return
    try:
        from google.cloud import secretmanager

        client = secretmanager.SecretManagerServiceClient()
        name = f"projects/{SECRET_PROJECT}/secrets/{WANDB_SECRET_NAME}/versions/latest"
        os.environ["WANDB_API_KEY"] = client.access_secret_version(name=name).payload.data.decode("utf-8")
        log.info("Loaded WANDB_API_KEY from Secret Manager")
    except Exception as e:
        log.warning(f"Could not load WANDB_API_KEY from Secret Manager: {e}")


def _make_loader(path: str, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    data = torch.load(path, weights_only=False)
    dataset = TensorDataset(data["images"], data["targets"])
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )


def _export_onnx(best_path: str) -> str:
    """Load the best checkpoint and export it to ONNX. Returns the ONNX file path."""
    checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
    model = Model()
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    onnx_path = best_path.replace(".ckpt", ".onnx")
    torch.onnx.export(
        model,
        (torch.randn(1, 3, 64, 64),),
        onnx_path,
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}},
    )
    log.info(f"Exported ONNX model to {onnx_path}")
    return onnx_path


def _upload_and_register(best_path: str, val_acc: float | None, run_id: str) -> None:
    """Export to ONNX, upload to GCS, and register in the Vertex Model Registry."""
    if not best_path:
        log.warning("No best_model_path available; skipping model registration.")
        return
    if val_acc is None:
        log.warning("No numeric val_acc available; skipping model registration.")
        return

    onnx_path = _export_onnx(best_path)

    artifact_dir = f"checkpoints/{run_id}"
    bucket = storage.Client().bucket(MODEL_BUCKET)
    bucket.blob(f"{artifact_dir}/model.onnx").upload_from_filename(onnx_path)

    artifact_uri = f"gs://{MODEL_BUCKET}/{artifact_dir}"
    log.info(f"Uploaded ONNX model to {artifact_uri}/model.onnx")

    vertex_registry.register_candidate(artifact_uri=artifact_uri, val_acc=val_acc)


@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def train(cfg: DictConfig) -> None:
    _ensure_wandb_key()
    pl.seed_everything(cfg.training.seed, workers=True)

    data_dir = Path(cfg.data_dir)
    num_workers = cfg.training.num_workers
    train_loader = _make_loader(str(data_dir / "train.pt"), cfg.training.batch_size, True, num_workers)
    val_loader = _make_loader(str(data_dir / "val.pt"), cfg.training.batch_size, False, num_workers)
    test_loader = _make_loader(str(data_dir / "test.pt"), cfg.training.batch_size, False, num_workers)

    model = Model(num_classes=cfg.model.num_classes, lr=cfg.training.lr)

    timestamp = datetime.now().strftime("%m%d-%H%M")
    run_name = cfg.wandb.name or (f"lr{cfg.training.lr}_bs{cfg.training.batch_size}_{timestamp}")

    wandb_logger = WandbLogger(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        name=run_name,
        log_model=False,
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.training.checkpoint_dir,
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        filename="best-{epoch:02d}-{val_loss:.3f}",
        verbose=True,
    )

    val_acc_tracker = ModelCheckpoint(
        dirpath=cfg.training.checkpoint_dir,
        monitor="val_acc",
        mode="max",
        save_top_k=1,
        filename="valacc-{epoch:02d}-{val_acc:.4f}",
        verbose=False,
    )
    early_stopping_callback = EarlyStopping(
        monitor="val_loss",
        patience=cfg.training.patience,
        mode="min",
        verbose=True,
    )

    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        callbacks=[checkpoint_callback, val_acc_tracker, early_stopping_callback],
        logger=wandb_logger,
        default_root_dir=cfg.training.checkpoint_dir,
        limit_train_batches=cfg.training.limit_train_batches,
        log_every_n_steps=cfg.training.log_every_n_steps,
    )

    trainer.fit(model, train_loader, val_loader)
    log.info(f"Best model saved: {checkpoint_callback.best_model_path}")

    best_val_acc_score = val_acc_tracker.best_model_score
    best_val_acc = float(best_val_acc_score) if best_val_acc_score is not None else None

    trainer.test(model, test_loader, ckpt_path="best")

    if cfg.training.register_model:
        run_id = wandb_logger.version or run_name
        _upload_and_register(checkpoint_callback.best_model_path, best_val_acc, run_id)


if __name__ == "__main__":
    train()
