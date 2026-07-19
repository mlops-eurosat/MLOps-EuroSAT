"""Evaluate a trained checkpoint on the test split.

Prints a per-class classification report:

    python src/mlops_eurosat/evaluate.py models/checkpoints/<name>.ckpt
"""

import torch
import typer
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, TensorDataset

from mlops_eurosat.model import Model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(checkpoint_path: str) -> Model:
    """Load a Lightning checkpoint into a `Model` in eval mode on DEVICE.

    Args:
        checkpoint_path: Path to a .ckpt file written during training.

    Returns:
        The model with the checkpoint weights loaded.
    """
    model = Model()

    checkpoint = torch.load(
        checkpoint_path,
        map_location=torch.device("cpu"),
    )

    model.load_state_dict(checkpoint["state_dict"])
    model.to(DEVICE)
    model.eval()

    return model


def predict(
    model: Model,
    dataloader: DataLoader,
) -> tuple[list[int], list[int]]:
    """Run the model over a dataloader without gradients.

    Args:
        model: Model in eval mode.
        dataloader: Batches of (images, targets).

    Returns:
        Two lists of class indices: (targets, predictions).
    """
    all_preds: list[int] = []
    all_targets: list[int] = []

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(DEVICE)
            targets = targets.to(DEVICE)

            logits = model(images)
            preds = logits.argmax(dim=1)

            all_preds.extend(preds.cpu().tolist())
            all_targets.extend(targets.cpu().tolist())

    return all_targets, all_preds


def evaluate(model_checkpoint: str) -> None:
    """CLI entry point: score a checkpoint on data/processed/test.pt.

    Args:
        model_checkpoint: Path to the .ckpt file to evaluate.
    """
    print(f"Evaluating {model_checkpoint}")

    model = load_model(model_checkpoint)

    data = torch.load("data/processed/test.pt")

    test_set = TensorDataset(
        data["images"],
        data["targets"],
    )

    test_loader = DataLoader(
        test_set,
        batch_size=32,
    )

    targets, preds = predict(
        model,
        test_loader,
    )

    print("\nClassification Report:")
    print(
        classification_report(
            targets,
            preds,
            target_names=data["classes"],
        )
    )


if __name__ == "__main__":
    typer.run(evaluate)
