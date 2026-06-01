from unittest.mock import MagicMock

import pytest
import torch

from mlops_eurosat.model import Model


@pytest.mark.parametrize("batch_size", [1, 8])
def test_model(batch_size: int) -> None:
    model = Model()
    x = torch.randn(batch_size, 3, 64, 64)
    y = model(x)

    assert y.shape == (batch_size, 10)


def test_training_step_returns_loss():
    model = Model()
    model.log = MagicMock()

    images = torch.randn(4, 3, 64, 64)
    targets = torch.tensor([0, 1, 2, 3])

    loss = model.training_step((images, targets))

    assert loss.ndim == 0


def test_backward_pass_computes_gradients():
    model = Model()
    model.log = MagicMock()  # mock logs so no warnings are thrown

    images = torch.randn(4, 3, 64, 64)
    targets = torch.tensor([0, 1, 2, 3])

    loss = model.training_step((images, targets))
    loss.backward()

    grads = [param.grad for param in model.parameters() if param.requires_grad]

    assert any(grad is not None for grad in grads)
