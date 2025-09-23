from pathlib import Path

import torch
from torch import nn, optim
from torchmetrics.functional import r2_score

from data_loading import get_dataloaders
from model import RNNModel
from utils import plot_loss, plot_preds_vs_targets, test, train


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cuda":
        torch.backends.cudnn.benchmark = True

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
        embed_dim=8,
        num_size=5,
        hidden_size=64,
        output_size=1,
        num_layers=2,
    ).to(device)

    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(rnn_model.parameters(), lr=0.001)
    epochs = 500
    patience = 100
    min_delta = 0.0

    train_results, val_results = train(
        rnn_model,
        train_dl,
        val_dl,
        loss_fn,
        r2_score,
        optimizer,
        epochs,
        patience,
        min_delta,
        device,
    )

    plot_loss(train_results["loss"], val_results["loss"])

    plot_preds_vs_targets(
        train_results["preds"], train_results["targets"], "Train preds vs targets"
    )
    plot_preds_vs_targets(
        val_results["preds"], val_results["targets"], "Validation preds vs targets"
    )

    test_results = test(rnn_model, test_dl, loss_fn, r2_score, device)
    plot_preds_vs_targets(
        test_results["preds"], test_results["targets"], "Test preds vs targets"
    )


if __name__ == "__main__":
    main()
