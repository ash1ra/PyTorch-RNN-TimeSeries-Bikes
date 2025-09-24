from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader


class RMSLELoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred: torch.Tensor, actual: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(self.mse(pred, actual))


def plot_preds_vs_targets(
    preds: list[np.ndarray], targets: list[np.ndarray], title: str
) -> None:
    plt.figure(figsize=(12, 6))
    plt.plot(preds, label="Predictions", color="red", alpha=0.7)
    plt.plot(targets, label="Targets", color="blue", alpha=0.7)
    plt.xlabel("Days")
    plt.ylabel("Count")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_loss(train_loss_list: list[float], val_loss_list: list[float]) -> None:
    epochs = list(range(1, len(train_loss_list) + 1))

    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    axs[0].plot(epochs, train_loss_list, label="Train loss")
    axs[0].set_xlabel("Epochs")
    axs[0].set_ylabel("Loss (MSE)")
    axs[0].legend()
    axs[0].grid(True)

    axs[1].plot(epochs, val_loss_list, label="Validation loss")
    axs[1].set_xlabel("Epochs")
    axs[1].set_ylabel("Loss (MSE)")
    axs[1].legend()
    axs[1].grid(True)

    fig.suptitle("Train and validation losses")
    plt.tight_layout()
    plt.show()


def train_step(
    model: nn.Module,
    train_dl: DataLoader,
    loss_fn: nn.Module,
    metric_fn: Callable,
    optimizer: optim.Optimizer,
    device: str = "cpu",
) -> tuple[float, float, list, list]:
    train_loss, train_metric = 0, 0
    train_preds, train_targets = [], []

    model.train()

    for cat_inputs, num_inputs, targets in train_dl:
        cat_inputs, num_inputs, targets = (
            cat_inputs.to(device),
            num_inputs.to(device),
            targets.to(device),
        )

        preds = model(cat_inputs, num_inputs)

        train_preds.append(preds.detach().cpu())
        train_targets.append(targets.detach().cpu())

        loss = loss_fn(preds, targets)

        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss /= len(train_dl)

    # train_preds = torch.cat(train_preds)
    # train_targets = torch.cat(train_targets)

    train_preds = torch.exp(torch.cat(train_preds))
    train_targets = torch.exp(torch.cat(train_targets))

    train_metric = metric_fn(train_preds, train_targets).item()

    return train_loss, train_metric, train_preds.numpy(), train_targets.numpy()


def test_step(
    model: nn.Module,
    test_dl: DataLoader,
    loss_fn: nn.Module,
    metric_fn: Callable,
    device: str = "cpu",
) -> tuple[float, float, list, list]:
    test_loss, test_metric = 0, 0
    test_preds, test_targets = [], []

    model.eval()

    with torch.inference_mode():
        for cat_inputs, num_inputs, targets in test_dl:
            cat_inputs, num_inputs, targets = (
                cat_inputs.to(device),
                num_inputs.to(device),
                targets.to(device),
            )

            preds = model(cat_inputs, num_inputs)

            test_preds.append(preds.detach().cpu())
            test_targets.append(targets.detach().cpu())

            loss = loss_fn(preds, targets)

            test_loss += loss.item()

    test_loss /= len(test_dl)

    # test_preds = torch.cat(test_preds)
    # test_targets = torch.cat(test_targets)

    test_preds = torch.exp(torch.cat(test_preds))
    test_targets = torch.exp(torch.cat(test_targets))

    test_metric = metric_fn(test_preds, test_targets).item()

    return test_loss, test_metric, test_preds.numpy(), test_targets.numpy()


def train(
    model: nn.Module,
    train_dl: DataLoader,
    val_dl: DataLoader,
    loss_fn: nn.Module,
    metric_fn: Callable,
    optimizer: optim.Optimizer,
    epochs: int,
    patience: int = 10,
    min_delta: float = 0.0,
    device: str = "cpu",
) -> tuple[dict[str, list], dict[str, list]]:
    train_results = {
        "loss": [],
        "metric": [],
        "preds": [],
        "targets": [],
    }
    val_results = {
        "loss": [],
        "metric": [],
        "preds": [],
        "targets": [],
    }
    best_val_loss = float("inf")
    patience_counter = 0
    temp_state_dict_path = Path("data/temp_best_state_dict.pt")

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=50,
        min_lr=1e-6,
    )

    for epoch in range(1, epochs + 1):
        train_loss, train_metric, train_preds, train_targets = train_step(
            model, train_dl, loss_fn, metric_fn, optimizer, device
        )
        val_loss, val_metric, val_preds, val_targets = test_step(
            model, val_dl, loss_fn, metric_fn, device
        )

        train_results["loss"].append(train_loss)
        train_results["metric"].append(train_metric)

        val_results["loss"].append(val_loss)
        val_results["metric"].append(val_metric)

        scheduler.step(val_loss)

        print(
            f"Epoch: {epoch} | Train loss: {train_loss:.2f} | Train {metric_fn.__name__}: {train_metric:.2f} | Val loss: {val_loss:.2f} | Val {metric_fn.__name__}: {val_metric:.2f}"
        )

        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            torch.save(model.state_dict(), temp_state_dict_path)
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch} epochs")
            break

    if best_val_loss < val_loss:
        model.load_state_dict(torch.load(temp_state_dict_path, map_location=device))
        print(f"Loaded best model parameters with val_loss={best_val_loss:.4f}")

    Path(temp_state_dict_path).unlink(missing_ok=True)

    _, _, train_results["preds"], train_results["targets"] = test_step(
        model, train_dl, loss_fn, metric_fn, device
    )
    _, _, val_results["preds"], val_results["targets"] = test_step(
        model, val_dl, loss_fn, metric_fn, device
    )

    return train_results, val_results


def test(
    model: nn.Module,
    data_loader: DataLoader,
    loss_fn: nn.Module,
    metric_fn: Callable,
    device: str = "cpu",
) -> dict[str, list]:
    test_loss, test_metric, test_preds, test_targets = test_step(
        model, data_loader, loss_fn, metric_fn, device
    )
    print(f"Test loss: {test_loss:.2f} | Test {metric_fn.__name__}: {test_metric:.2f}")

    return {
        "loss": test_loss,
        "metric": test_metric,
        "preds": test_preds,
        "targets": test_targets,
    }
