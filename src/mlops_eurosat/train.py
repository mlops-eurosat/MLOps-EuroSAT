import logging
from datetime import datetime
from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, TensorDataset

from mlops_eurosat.model import Model

log = logging.getLogger(__name__)

MODEL_ARTIFACT_NAME = "eurosat-classifier"


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


def _log_model_artifact(
    checkpoint_callback: ModelCheckpoint,
    metrics: dict,
    data_dir: Path,
) -> None:
    """Log the best (val_loss) checkpoint under a single shared collection name.

    Carries val_acc/test_acc/val_loss plus the normalisation stats (mean/std)
    and class order in the metadata, so that:
      * the registry promotion can compare runs (champion/challenger), and
      * the inference API is self-contained (no dependency on the .pt files).
    """
    best_path = checkpoint_callback.best_model_path
    if not best_path:
        log.warning("No best_model_path available; skipping model artifact log.")
        return

    def _num(key: str):
        v = metrics.get(key)
        if v is None:
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    metadata: dict = {}
    for key in ("val_acc", "test_acc", "val_loss"):
        val = _num(key)
        if val is not None:
            metadata[key] = val

    norm = torch.load(data_dir / "train.pt", weights_only=False)
    metadata["mean"] = norm["mean"].tolist()
    metadata["std"] = norm["std"].tolist()
    metadata["classes"] = norm["classes"]

    artifact = wandb.Artifact(
        name=MODEL_ARTIFACT_NAME,
        type="model",
        metadata=metadata,
    )
    artifact.add_file(best_path)
    wandb.log_artifact(artifact)
    log.info(f"Logged model artifact '{MODEL_ARTIFACT_NAME}' with metadata keys={list(metadata)}")


@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def train(cfg: DictConfig) -> None:
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

    metrics: dict[str, object] = dict(trainer.callback_metrics)
    if best_val_acc is not None:
        metrics["val_acc"] = best_val_acc

    _log_model_artifact(checkpoint_callback, metrics, data_dir)


if __name__ == "__main__":
    train()
