import torch
from torch.utils.data import DataLoader, TensorDataset

from mlops_eurosat.evaluate import load_model, predict
from mlops_eurosat.model import Model


def test_load_model_loads_state_dict(tmp_path):
    model = Model()

    # create a valid checkpoint
    checkpoint_path = tmp_path / "ckpt.pt"
    torch.save(
        {"state_dict": model.state_dict()},
        checkpoint_path,
    )

    loaded_model = load_model(str(checkpoint_path))

    assert isinstance(loaded_model, Model)


class DummyModel:
    def __call__(self, x):
        return torch.tensor(
            [
                [10.0, 0.0],
                [0.0, 10.0],
            ]
        )


def test_predict_returns_targets_and_predictions():
    dataset = TensorDataset(
        torch.randn(2, 3, 64, 64),
        torch.tensor([0, 1]),
    )

    loader = DataLoader(dataset, batch_size=2)

    targets, preds = predict(
        DummyModel(),
        loader,
    )

    assert targets == [0, 1]
    assert preds == [0, 1]
