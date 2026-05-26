import torch
from torch.utils.data import DataLoader, Subset, TensorDataset

from mlops_eurosat.model import Model


def train():

    print("Loading processed dataset...")

    data = torch.load("data/processed/train.pt")

    images = data["images"]
    labels = data["targets"]
    classes = data["classes"]

    print(f"Images shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Classes: {classes}")

    dataset = TensorDataset(images, labels)

    print(f"Full dataset size: {len(dataset)}")

    # Small subset for testing
    subset_size = 2000

    random_indices = torch.randperm(len(dataset))[:subset_size]

    subset = Subset(
        dataset,
        random_indices.tolist(),
    )

    dataloader = DataLoader(
        subset,
        batch_size=32,
        shuffle=True,
    )

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    print(f"Using device: {device}")

    model = Model().to(device)

    criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
    )

    model.train()

    for epoch in range(10):
        print(f"\nEpoch {epoch + 1}")

        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            print(f"Loss: {loss.item():.4f}")


if __name__ == "__main__":
    train()
