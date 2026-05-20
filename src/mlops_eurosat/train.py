from mlops_eurosat.data import MyDataset
from mlops_eurosat.model import Model


def train():
    dataset = MyDataset("data/raw")
    model = Model()
    # add rest of your training code here


if __name__ == "__main__":
    train()
