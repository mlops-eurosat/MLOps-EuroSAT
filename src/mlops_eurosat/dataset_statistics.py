from pathlib import Path

import matplotlib.pyplot as plt
import torch
import typer


def _load_split(processed_dir: Path, name: str) -> dict:
    """Load a processed split dict: {images, targets, classes, mean, std}."""
    return torch.load(processed_dir / f"{name}.pt", weights_only=False)


def _denormalize(images: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Undo per-channel normalization so images are viewable in [0, 1]."""
    mean = mean.view(1, 3, 1, 1)
    std = std.view(1, 3, 1, 1)
    return (images * std + mean).clamp(0, 1)


def dataset_statistics(datadir: str = "data/processed") -> None:
    """Compute dataset statistics for the EuroSAT processed splits."""
    processed_dir = Path(datadir)
    reports_dir = Path("reports/figures")
    reports_dir.mkdir(parents=True, exist_ok=True)

    splits = {}
    for name in ("train", "val", "test"):
        path = processed_dir / f"{name}.pt"
        if not path.exists():
            print(f"[warn] {path} not found, skipping")
            continue
        splits[name] = _load_split(processed_dir, name)

    if not splits:
        raise FileNotFoundError(f"No processed splits found in {processed_dir}")

    classes = splits[next(iter(splits))]["classes"]
    num_classes = len(classes)

    for name, payload in splits.items():
        images = payload["images"]
        targets = payload["targets"]
        print(f"\n=== {name.upper()} dataset ===")
        print(f"Number of images: {len(images)}")
        print(f"Image shape:      {tuple(images[0].shape)}")
        print(f"Pixel range:      [{images.min():.3f}, {images.max():.3f}]")
        dist = torch.bincount(targets, minlength=num_classes)
        print("Label distribution:")
        for cls_name, count in zip(classes, dist.tolist()):
            print(f"  {cls_name:<22} {count}")

    # Figure 1: grid of de-normalized sample images from the train split
    train = splits.get("train", splits[next(iter(splits))])
    images = _denormalize(train["images"], train["mean"], train["std"])
    targets = train["targets"]
    n = min(25, len(images))
    fig, axes = plt.subplots(5, 5, figsize=(10, 10))
    for i, ax in enumerate(axes.flat):
        if i < n:
            ax.imshow(images[i].permute(1, 2, 0).cpu())
            ax.set_title(classes[targets[i]], fontsize=7)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(reports_dir / "eurosat_images.png")
    plt.close(fig)
    print(f"\nSaved {reports_dir / 'eurosat_images.png'}")

    # Figures 2+: label distribution per split
    for name, payload in splits.items():
        dist = torch.bincount(payload["targets"], minlength=num_classes)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(torch.arange(num_classes), dist)
        ax.set_title(f"{name.capitalize()} label distribution")
        ax.set_xlabel("Label")
        ax.set_ylabel("Count")
        ax.set_xticks(torch.arange(num_classes))
        ax.set_xticklabels(classes, rotation=45, ha="right")
        fig.tight_layout()
        fig.savefig(reports_dir / f"{name}_label_distribution.png")
        plt.close(fig)
        print(f"Saved {reports_dir / f'{name}_label_distribution.png'}")


if __name__ == "__main__":
    typer.run(dataset_statistics)
