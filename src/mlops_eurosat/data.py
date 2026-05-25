from pathlib import Path

import torch
import typer
from PIL import Image
from torch.utils.data import Dataset


class MyDataset(Dataset):
    """EuroSAT RGB dataset."""

    def __init__(self, data_path: Path) -> None:
        self.data_path = data_path

        self.images = None
        self.targets = None
        self.classes = None

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.targets)

    def __getitem__(self, index: int):
        """Return a given sample from the dataset."""
        return self.images[index], self.targets[index]

    def preprocess(self, output_folder: Path) -> None:
        """Preprocess the raw data and save it to the output folder."""
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)

        classes = sorted([p.name for p in self.data_path.iterdir() if p.is_dir()])

        class_to_idx = {class_name: idx for idx, class_name in enumerate(classes)}

        images = []
        targets = []

        for class_name in classes:
            class_folder = self.data_path / class_name

            for image_path in sorted(class_folder.glob("*.jpg")):
                image = Image.open(image_path).convert("RGB")
                image = torch.tensor(list(image.getdata()), dtype=torch.float32)
                image = image.reshape(64, 64, 3).permute(2, 0, 1) / 255.0

                images.append(image)
                targets.append(class_to_idx[class_name])

        images = torch.stack(images)
        targets = torch.tensor(targets, dtype=torch.long)

        generator = torch.Generator().manual_seed(42)
        indices = torch.randperm(len(images), generator=generator)

        split_idx = int(0.8 * len(indices))
        train_idx = indices[:split_idx]
        test_idx = indices[split_idx:]

        train_data = {
            "images": images[train_idx],
            "targets": targets[train_idx],
            "classes": classes,
        }

        test_data = {
            "images": images[test_idx],
            "targets": targets[test_idx],
            "classes": classes,
        }

        torch.save(train_data, output_folder / "train.pt")
        torch.save(test_data, output_folder / "test.pt")


def preprocess(data_path: Path, output_folder: Path) -> None:
    print("Preprocessing data...")
    dataset = MyDataset(data_path)
    dataset.data_path = Path(data_path)
    dataset.preprocess(output_folder)


if __name__ == "__main__":
    typer.run(preprocess)
