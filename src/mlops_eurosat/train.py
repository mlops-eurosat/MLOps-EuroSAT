import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader, Subset, TensorDataset

from mlops_eurosat.model import Model


def train():
    # Load data
    data = torch.load("data/processed/train.pt")
    dataset = TensorDataset(data["images"], data["targets"])

    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Model
    model = Model()

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath="./models",
        monitor="train_loss",
        mode="min",
        save_top_k=1,  # only keep best model
        verbose=True,
    )
    early_stopping_callback = EarlyStopping(
        monitor="train_loss",
        patience=3,  # stops after 3 epochs without improvement
        mode="min",
        verbose=True,
    )

    # Trainer
    trainer = pl.Trainer(
        max_epochs=10,
        callbacks=[checkpoint_callback, early_stopping_callback],
        default_root_dir="./models",
        limit_train_batches=0.2,
    )

    trainer.fit(model, dataloader)
    print(f"Bestes Modell gespeichert unter: {checkpoint_callback.best_model_path}")


if __name__ == "__main__":
    train()
