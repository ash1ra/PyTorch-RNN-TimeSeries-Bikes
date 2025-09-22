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

    # input_size = train_dl.dataset.data[0].size
    cat_sizes = [2, 12, 7, 2, 2, 4, 3]
    rnn_model = RNNModel(
        cat_sizes,
        embed_dim=4,
        num_size=5,
        hidden_size=64,
        output_size=1,
        num_layers=2,
    )

    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(rnn_model.parameters(), lr=0.001)
    epochs = 100
    patience = 50
    min_delta = 0.0

    train(
        rnn_model,
        train_dl,
        val_dl,
        loss_fn,
        mape_metric,
        optimizer,
        epochs,
        patience,
        min_delta,
    )
    test(rnn_model, test_dl, loss_fn, mape_metric)


if __name__ == "__main__":
    main()
