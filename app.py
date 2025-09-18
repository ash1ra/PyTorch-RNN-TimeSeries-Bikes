from pathlib import Path

from torch import nn, optim

from data_loading import get_dataloaders
from model import RNNModel
from utils import test, train, mape_metric


def main() -> None:
    path = Path("data")

    train_ds_path = path / "train_ds.pt"
    val_ds_path = path / "val_ds.pt"
    test_ds_path = path / "test_ds.pt"

    train_dl, val_dl, test_dl = get_dataloaders(
        train_ds_path, val_ds_path, test_ds_path, 32
    )

    input_size = train_dl.dataset.data[0].size
    rnn_model = RNNModel(input_size, 1, 1)

    epochs = 100
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(rnn_model.parameters(), lr=0.1)

    train(rnn_model, train_dl, val_dl, loss_fn, mape_metric, optimizer, epochs)
    test(rnn_model, test_dl, loss_fn, mape_metric)


if __name__ == "__main__":
    main()
