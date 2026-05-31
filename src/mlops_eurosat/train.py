import logging

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader, TensorDataset

from mlops_eurosat.model import Model

log = logging.getLogger(__name__)


@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def train(cfg: DictConfig):
    pl.seed_everything(cfg.training.seed, workers=True)

    data = torch.load(cfg.data_path)
    dataset = TensorDataset(data["images"], data["targets"])
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
    )

    model = Model(num_classes=cfg.model.num_classes, lr=cfg.training.lr)

    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.training.checkpoint_dir,
        monitor="train_loss",
        mode="min",
        save_top_k=1,
        verbose=True,
    )
    early_stopping_callback = EarlyStopping(
        monitor="train_loss",
        patience=cfg.training.patience,
        mode="min",
        verbose=True,
    )

    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        callbacks=[checkpoint_callback, early_stopping_callback],
        default_root_dir=cfg.training.checkpoint_dir,
        limit_train_batches=cfg.training.limit_train_batches,
    )
    trainer.fit(model, dataloader)

    log.info(f"Best model saved: {checkpoint_callback.best_model_path}")


if __name__ == "__main__":
    train()