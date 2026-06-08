import torch
import typer
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, TensorDataset

from mlops_eurosat.model import Model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(checkpoint_path: str) -> Model:
    """Load a trained model from a checkpoint."""
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
    """Generate predictions for a dataloader."""
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
    """Evaluate a trained model."""
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
