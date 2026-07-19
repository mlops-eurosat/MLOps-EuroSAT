"""CNN classifier for EuroSAT, implemented as a PyTorch Lightning module."""

import pytorch_lightning as pl
import torch
from torch import nn
from torchmetrics.classification import MulticlassAccuracy


class Model(pl.LightningModule):
    """CNN for EuroSAT image classification.

    Four Conv-BatchNorm-ReLU-MaxPool blocks (32 to 256 channels), global
    average pooling, then a two-layer classifier with dropout. Trained with
    cross-entropy; accuracy is tracked per split.

    Args:
        num_classes: Number of output classes.
        lr: Initial learning rate for Adam.
    """

    def __init__(self, num_classes: int = 10, lr: float = 1e-3) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr

        self.features = nn.Sequential(
            self._block(3, 32),
            self._block(32, 64),
            self._block(64, 128),
            self._block(128, 256),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

        self.loss_fn = nn.CrossEntropyLoss()
        self.train_acc = MulticlassAccuracy(num_classes=num_classes)
        self.val_acc = MulticlassAccuracy(num_classes=num_classes)
        self.test_acc = MulticlassAccuracy(num_classes=num_classes)

    @staticmethod
    def _block(in_ch: int, out_ch: int) -> nn.Sequential:
        # Conv -> BatchNorm -> ReLU -> MaxPool
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map an image batch (B, 3, 64, 64) to logits (B, num_classes)."""
        x = self.features(x)
        x = self.classifier(x)
        return x

    def _shared_step(self, batch):
        imgs, targets = batch
        preds = self(imgs)
        loss = self.loss_fn(preds, targets)
        return loss, preds, targets

    def training_step(self, batch):
        """Compute the loss on a training batch and log train_loss/train_acc."""
        loss, preds, targets = self._shared_step(batch)
        self.train_acc(preds, targets)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", self.train_acc, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch):
        """Compute the loss on a validation batch and log val_loss/val_acc."""
        loss, preds, targets = self._shared_step(batch)
        self.val_acc(preds, targets)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.val_acc, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def test_step(self, batch):
        """Compute the loss on a test batch and log test_loss/test_acc."""
        loss, preds, targets = self._shared_step(batch)
        self.test_acc(preds, targets)
        self.log("test_loss", loss)
        self.log("test_acc", self.test_acc, on_epoch=True, on_step=False)
        return loss

    def configure_optimizers(self):
        """Adam plus ReduceLROnPlateau (halved after 3 epochs without val_loss improvement)."""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
        }


if __name__ == "__main__":
    model = Model()
    print(model)
    x = torch.randn(1, 3, 64, 64)
    y = model(x)
    print(f"Output shape: {y.shape}")
