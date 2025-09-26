from pathlib import Path

import torch
from torch import optim
from torchinfo import summary
from torchmetrics.functional import r2_score

from data_loading import get_dataloaders
from model import RNNModel

# from one_hot_data_loading import get_dataloaders
from utils import RMSELoss, plot_loss, plot_preds_vs_targets, test, train

BATCH_SIZE = 32


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

    # input_size = 21
    cat_sizes = [2, 2, 2, 4, 4]

    rnn_model = RNNModel(
        cat_sizes,
        num_size=9,
        # input_size=input_size,
        hidden_size=16,
        output_size=1,
        num_layers=1,
    ).to(device)

    loss_fn = RMSELoss()
    optimizer = optim.Adam(rnn_model.parameters(), lr=0.001, weight_decay=1e-4)
    epochs = 500
    patience = 20
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

    test_results = test(rnn_model, test_dl, loss_fn, r2_score, device)

    # summary(rnn_model, input_size=(BATCH_SIZE, 24, 15))

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
