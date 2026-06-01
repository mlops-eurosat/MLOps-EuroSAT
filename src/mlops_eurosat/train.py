import logging
from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader, TensorDataset

from mlops_eurosat.model import Model

log = logging.getLogger(__name__)


def _make_loader(path: str, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    data = torch.load(path)
    dataset = TensorDataset(data["images"], data["targets"])
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )


@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def train(cfg: DictConfig) -> None:
    pl.seed_everything(cfg.training.seed, workers=True)

    data_dir = Path(cfg.data_dir)
    num_workers = cfg.training.num_workers

    train_loader = _make_loader(str(data_dir / "train.pt"), cfg.training.batch_size, True, num_workers)
    val_loader = _make_loader(str(data_dir / "val.pt"), cfg.training.batch_size, False, num_workers)
    test_loader = _make_loader(str(data_dir / "test.pt"), cfg.training.batch_size, False, num_workers)

    model = Model(num_classes=cfg.model.num_classes, lr=cfg.training.lr)

    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.training.checkpoint_dir,
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        filename="best-{epoch:02d}-{val_loss:.3f}",
        verbose=True,
    )
    early_stopping_callback = EarlyStopping(
        monitor="val_loss",
        patience=cfg.training.patience,
        mode="min",
        verbose=True,
    )

    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        callbacks=[checkpoint_callback, early_stopping_callback],
        default_root_dir=cfg.training.checkpoint_dir,
        limit_train_batches=cfg.training.limit_train_batches,
        log_every_n_steps=cfg.training.log_every_n_steps,
    )

    trainer.fit(model, train_loader, val_loader)
    log.info(f"Best model saved: {checkpoint_callback.best_model_path}")

    trainer.test(model, test_loader, ckpt_path="best")


if __name__ == "__main__":
    train()
