import torch
import typer
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.utils.data import TensorDataset

from mlops_eurosat.model import Model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def evaluate(model_checkpoint: str) -> None:
    """Evaluate a trained model."""
    print(f"Evaluating {model_checkpoint}")

    model = Model()
    checkpoint = torch.load(model_checkpoint, map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)

    data = torch.load("data/processed/test.pt")

    test_set = TensorDataset(data["images"], data["targets"])
    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=32)

    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for img, target in test_dataloader:
            img, target = img.to(DEVICE), target.to(DEVICE)

            logits = model(img)
            preds = logits.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    accuracy = accuracy_score(all_targets, all_preds)
    macro_f1 = f1_score(all_targets, all_preds, average="macro")

    print("\nClassification Report:")
    print(
        classification_report(
            all_targets,
            all_preds,
            target_names=data["classes"],
        )
    )


if __name__ == "__main__":
    typer.run(evaluate)
