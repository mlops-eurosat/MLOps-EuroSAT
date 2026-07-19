"""EuroSAT preprocessing: raw JPEGs to normalised train/val/test tensors.

Run directly or through DVC (`dvc repro preprocess`):

    python src/mlops_eurosat/data.py data/raw/EuroSAT_RGB data/processed
"""

from pathlib import Path

import numpy as np
import torch
import typer
from PIL import Image
from torch.utils.data import Dataset


def stratified_split(
    targets: torch.Tensor,
    fracs: tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split sample indices into train/val/test, keeping the class balance.

    Args:
        targets: Class label per sample, shape (N,).
        fracs: Train/val/test fractions; must sum to 1.
        seed: Seed for the per-class shuffle.

    Returns:
        Three index tensors (train, val, test) into `targets`.
    """
    assert abs(sum(fracs) - 1.0) < 1e-6, "fracs must sum to 1.0"
    generator = torch.Generator().manual_seed(seed)
    train_idx, val_idx, test_idx = [], [], []
    for class_id in targets.unique():
        idx = (targets == class_id).nonzero(as_tuple=True)[0]
        idx = idx[torch.randperm(len(idx), generator=generator)]
        n_train = int(fracs[0] * len(idx))
        n_val = int((fracs[0] + fracs[1]) * len(idx))
        train_idx.append(idx[:n_train])
        val_idx.append(idx[n_train:n_val])
        test_idx.append(idx[n_val:])
    return torch.cat(train_idx), torch.cat(val_idx), torch.cat(test_idx)


class MyDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """EuroSAT RGB dataset, read from a folder with one subfolder per class."""

    def __init__(self, data_path: Path) -> None:
        self.data_path = Path(data_path)
        self.images: torch.Tensor = torch.empty(0)
        self.targets: torch.Tensor = torch.empty(0, dtype=torch.long)
        self.classes: list[str] = []

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.targets)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the (image, target) pair at `index`."""
        return self.images[index], self.targets[index]

    def preprocess(self, output_folder: Path) -> None:
        """Load the raw JPEGs, split and normalise them, and write the .pt files.

        Images are scaled to [0, 1], split with `stratified_split` and
        normalised with the mean/std of the train split only.

        Args:
            output_folder: Target folder for train.pt, val.pt and test.pt. Each
                file holds `images` (N, 3, 64, 64), `targets`, `classes` and the
                `mean`/`std` used for normalisation.
        """
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)

        classes = sorted(p.name for p in self.data_path.iterdir() if p.is_dir())
        class_to_idx = {class_name: idx for idx, class_name in enumerate(classes)}

        image_list: list[torch.Tensor] = []
        target_list: list[int] = []

        for class_name in classes:
            class_folder = self.data_path / class_name
            for image_path in sorted(class_folder.glob("*.jpg")):
                with Image.open(image_path) as pil_image:
                    rgb_image = pil_image.convert("RGB")
                    array = np.asarray(rgb_image, dtype=np.float32)  # H, W, C
                image_tensor = torch.from_numpy(array).permute(2, 0, 1) / 255.0
                image_list.append(image_tensor)
                target_list.append(class_to_idx[class_name])

        images = torch.stack(image_list)
        targets = torch.tensor(target_list, dtype=torch.long)

        # Stratified train / val / test split
        train_idx, val_idx, test_idx = stratified_split(targets, seed=42)

        # Normalisation
        train_images = images[train_idx]
        mean = train_images.mean(dim=(0, 2, 3)).view(1, 3, 1, 1)  # per channel
        std = train_images.std(dim=(0, 2, 3)).view(1, 3, 1, 1)

        for name, idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
            payload = {
                "images": (images[idx] - mean) / std,
                "targets": targets[idx],
                "classes": classes,
                "mean": mean.squeeze(),
                "std": std.squeeze(),
            }
            torch.save(payload, output_folder / f"{name}.pt")
            print(f"Saved {name}.pt: {len(idx)} samples")


def preprocess(data_path: Path, output_folder: Path) -> None:
    """CLI entry point: preprocess the raw EuroSAT folder into `output_folder`.

    Args:
        data_path: Raw data folder with one subfolder of JPEGs per class.
        output_folder: Where the processed train/val/test tensors are written.
    """
    print("Preprocessing data...")
    dataset = MyDataset(data_path)
    dataset.preprocess(output_folder)


if __name__ == "__main__":
    typer.run(preprocess)
