from pathlib import Path

import torch
from torch import nn, optim
from torchinfo import summary
from torchmetrics.functional import r2_score

from data_loading import get_dataloaders
from model import RNNModel
from utils import plot_loss, plot_preds_vs_targets, test, train

BATCH_SIZE = 16


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cuda":
        torch.backends.cudnn.benchmark = True

    path = Path("data")

    train_ds_path = path / "train_ds.pt"
    val_ds_path = path / "val_ds.pt"
    test_ds_path = path / "test_ds.pt"

    train_dl, val_dl, test_dl = get_dataloaders(
        train_ds_path, val_ds_path, test_ds_path, BATCH_SIZE
    )

    cat_sizes = [2, 2, 2, 4, 4]

    rnn_model = RNNModel(
        cat_sizes,
        num_size=12,
        hidden_size=64,
        output_size=1,
        num_layers=2,
        dropout=0.3,
        bidirectional=False,
    ).to(device)

    loss_fn = nn.L1Loss()
    optimizer = optim.Adam(rnn_model.parameters(), lr=0.001, weight_decay=1e-3)
    epochs = 500
    patience = 20
    min_delta = 0.0

    cat_x = torch.zeros((BATCH_SIZE, 72, 5), dtype=torch.int64, device=device)
    num_x = torch.zeros((BATCH_SIZE, 72, 12), dtype=torch.float32, device=device)

    summary(rnn_model, input_data=[cat_x, num_x])

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

    test_results = test(rnn_model, test_dl, loss_fn, r2_score, device)

    plot_loss(train_results["loss"], val_results["loss"])

    plot_preds_vs_targets(
        train_results["preds"], train_results["targets"], "Train preds vs targets"
    )
    plot_preds_vs_targets(
        val_results["preds"], val_results["targets"], "Validation preds vs targets"
    )

    plot_preds_vs_targets(
        test_results["preds"], test_results["targets"], "Test preds vs targets"
    )


if __name__ == "__main__":
    main()
