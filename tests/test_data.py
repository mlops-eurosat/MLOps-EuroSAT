import os

import pytest
import torch

from mlops_eurosat.data import stratified_split


def test_stratified_split_preserves_class_distribution():
    targets = torch.tensor([0] * 100 + [1] * 100)

    train_idx, val_idx, test_idx = stratified_split(targets)

    train_targets = targets[train_idx]
    val_targets = targets[val_idx]
    test_targets = targets[test_idx]

    assert (train_targets == 0).sum() == 70
    assert (train_targets == 1).sum() == 70

    assert (val_targets == 0).sum() == 15
    assert (val_targets == 1).sum() == 15

    assert (test_targets == 0).sum() == 15
    assert (test_targets == 1).sum() == 15


def test_stratified_split_uses_all_samples():
    targets = torch.tensor([0] * 100 + [1] * 100)

    train_idx, val_idx, test_idx = stratified_split(targets)

    assigned = torch.cat([train_idx, val_idx, test_idx])

    assert len(assigned) == len(targets)
    assert len(torch.unique(assigned)) == len(targets)


def test_stratified_split_has_no_overlap():
    targets = torch.tensor([0] * 100 + [1] * 100)

    train_idx, val_idx, test_idx = stratified_split(targets)

    train = set(train_idx.tolist())
    val = set(val_idx.tolist())
    test = set(test_idx.tolist())

    assert train.isdisjoint(val)
    assert train.isdisjoint(test)
    assert val.isdisjoint(test)


@pytest.mark.skipif(not os.path.exists("data/processed/test.pt"), reason="Data files not found")
def test_data():
    test = torch.load("data/processed/test.pt")
    assert len(test) == 5
    assert set(test.keys()) == set(["images", "targets", "classes", "mean", "std"])
    assert test["images"][0].shape == (3, 64, 64)
