"""Unit tests for the model (mlops_eurosat.model.Model).

Covers the forward pass, the Lightning train/val/test steps, gradient flow,
and optimizer config. All tests use small random tensors so they run fast and
need no real data.
"""

from unittest.mock import MagicMock

import pytest
import torch

from mlops_eurosat.model import Model


@pytest.mark.parametrize("batch_size", [1, 8])
def test_model(batch_size: int) -> None:
    """Forward pass outputs logits of shape (batch, 10) for the 10 classes.

    Parametrized over batch sizes to catch batch-dimension bugs (e.g. a layer
    that only works for one fixed size).
    """
    model = Model()
    x = torch.randn(batch_size, 3, 64, 64)
    y = model(x)

    assert y.shape == (batch_size, 10)


def test_training_step_returns_loss():
    """training_step must return a scalar loss (ndim == 0)."""
    model = Model()
    model.log = MagicMock()  # stub Lightning's log(); needs a Trainer otherwise

    images = torch.randn(4, 3, 64, 64)
    targets = torch.tensor([0, 1, 2, 3])

    loss = model.training_step((images, targets))

    assert loss.ndim == 0


def test_backward_pass_computes_gradients():
    """backward() on the loss must populate gradients on trainable params.

    Confirms the autograd graph is connected end-to-end, i.e. the model can
    actually learn. A broken/detached layer would leave all grads as None.
    """
    model = Model()
    model.log = MagicMock()  # mock logs so no warnings are thrown

    images = torch.randn(4, 3, 64, 64)
    targets = torch.tensor([0, 1, 2, 3])

    loss = model.training_step((images, targets))
    loss.backward()

    grads = [param.grad for param in model.parameters() if param.requires_grad]

    assert any(grad is not None for grad in grads)


def test_validation_step_returns_loss():
    """validation_step must return a scalar loss (same contract as training)."""
    model = Model()
    model.log = MagicMock()  # same reason: avoid the "no Trainer attached" error
    loss = model.validation_step((torch.randn(4, 3, 64, 64), torch.tensor([0, 1, 2, 3])))
    assert loss.ndim == 0


def test_test_step_returns_loss():
    """test_step must also return a scalar loss."""
    model = Model()
    model.log = MagicMock()
    loss = model.test_step((torch.randn(4, 3, 64, 64), torch.tensor([0, 1, 2, 3])))
    assert loss.ndim == 0


def test_configure_optimizers_returns_adam():
    """configure_optimizers() returns a dict whose optimizer is Adam."""
    model = Model()
    config = model.configure_optimizers()
    assert isinstance(config["optimizer"], torch.optim.Adam)
