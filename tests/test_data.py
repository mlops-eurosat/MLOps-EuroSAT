"""Unit tests for the data layer (mlops_eurosat.data).

Covers the stratified train/val/test split, the processed-data file format,
and the Dataset class. The split tests use a tiny synthetic label set so they
run fast and deterministically; the file-based test is skipped if the data
isn't present.
"""

import os
from pathlib import Path  # noqa: F401  (handy when extending the file-based tests)

import pytest
import torch

from mlops_eurosat.data import MyDataset, stratified_split


def test_stratified_split_preserves_class_distribution():
    """Each class is split into train/val/test by the given fractions, with no
    samples lost to rounding."""
    targets = torch.tensor([0] * 100 + [1] * 100)
    fracs = (0.7, 0.15, 0.15)
    train_idx, val_idx, test_idx = stratified_split(targets, fracs=fracs)

    for class_id in [0, 1]:
        n = (targets == class_id).sum().item()  # samples in this class

        assert (targets[train_idx] == class_id).sum() == int(fracs[0] * n)
        assert (targets[val_idx] == class_id).sum() == int((fracs[0] + fracs[1]) * n) - int(fracs[0] * n)
        assert (targets[test_idx] == class_id).sum() == n - int((fracs[0] + fracs[1]) * n)


def test_stratified_split_uses_all_samples():
    """No sample is dropped or duplicated across the three splits."""
    targets = torch.tensor([0] * 100 + [1] * 100)
    train_idx, val_idx, test_idx = stratified_split(targets)

    assigned = torch.cat([train_idx, val_idx, test_idx])
    assert len(assigned) == len(targets)  # every sample assigned
    assert len(torch.unique(assigned)) == len(targets)  # none assigned twice


def test_stratified_split_has_no_overlap():
    """The three splits are mutually disjoint (no index in two splits)."""
    targets = torch.tensor([0] * 100 + [1] * 100)
    train_idx, val_idx, test_idx = stratified_split(targets)

    train = set(train_idx.tolist())
    val = set(val_idx.tolist())
    test = set(test_idx.tolist())

    assert train.isdisjoint(val)
    assert train.isdisjoint(test)
    assert val.isdisjoint(test)


@pytest.mark.skipif(
    not os.path.exists("data/processed/test.pt"),
    reason="Data files not found",
)
def test_data():
    """Processed test.pt has the expected keys, length, and image shape."""
    test = torch.load("data/processed/test.pt")
    assert len(test) == 5
    assert set(test.keys()) == {"images", "targets", "classes", "mean", "std"}
    assert test["images"][0].shape == (3, 64, 64)


def test_mydataset_len_and_getitem(tmp_path):
    """Dataset reports correct length and returns (image, target) by index."""
    ds = MyDataset(tmp_path)  # path unused here; we inject tensors directly
    ds.images = torch.randn(5, 3, 64, 64)
    ds.targets = torch.tensor([0, 1, 2, 3, 4])

    assert len(ds) == 5

    img, target = ds[2]
    assert img.shape == (3, 64, 64)
    assert target == 2
