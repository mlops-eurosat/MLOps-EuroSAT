import pytest
import torch

from mlops_eurosat.train import _make_loader


@pytest.fixture
def data_file(tmp_path):
    """Write a tiny {images, targets} .pt file and return its path."""
    path = tmp_path / "data.pt"
    torch.save(
        {
            "images": torch.randn(8, 3, 64, 64),
            "targets": torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]),
        },
        path,
    )
    return str(path)


def test_make_loader_returns_correct_batch_shape(data_file):
    """A batch yields images of shape (B, 3, 64, 64) and matching targets."""
    loader = _make_loader(data_file, batch_size=4, shuffle=False, num_workers=0)
    images, targets = next(iter(loader))
    assert images.shape == (4, 3, 64, 64)
    assert targets.shape == (4,)


def test_make_loader_respects_batch_size(data_file):
    """The DataLoader batches with the requested batch_size."""
    loader = _make_loader(data_file, batch_size=2, shuffle=False, num_workers=0)
    images, _ = next(iter(loader))
    assert images.shape[0] == 2


def test_make_loader_covers_all_samples(data_file):
    """Every sample appears exactly once across all batches (none dropped)."""
    loader = _make_loader(data_file, batch_size=3, shuffle=False, num_workers=0)
    total = sum(images.shape[0] for images, _ in loader)
    assert total == 8  # 8 samples in the fixture
