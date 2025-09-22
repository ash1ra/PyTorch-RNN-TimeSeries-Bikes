from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader


def mape_metric(preds, targets):
    preds = np.exp(preds.detach().cpu().numpy())
    targets = np.exp(targets.detach().cpu().numpy())
    return np.mean(np.abs((targets - preds) / targets)) * 100


def train_step(
    model: nn.Module,
    train_dl: DataLoader,
    loss_fn: nn.Module,
    metric_fn: Callable,
    optimizer: optim.Optimizer,
    device: str = "cpu",
) -> tuple[float]:
    train_loss, train_metric = 0, 0
    model.train()

    for cat_inputs, num_inputs, targets in train_dl:
        cat_inputs, num_inputs, targets = (
            cat_inputs.to(device),
            num_inputs.to(device),
            targets.to(device),
        )

        preds = model(cat_inputs, num_inputs)

        loss = loss_fn(preds, targets)

        train_loss += loss.item()
        train_metric += metric_fn(preds, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss /= len(train_dl)
    train_metric /= len(train_dl)

    return train_loss, train_metric


def test_step(
    model: nn.Module,
    test_dl: DataLoader,
    loss_fn: nn.Module,
    metric_fn: Callable,
    device: str = "cpu",
) -> tuple[float]:
    test_loss, test_metric = 0, 0
    model.eval()

    with torch.inference_mode():
        for cat_inputs, num_inputs, targets in test_dl:
            cat_inputs, num_inputs, targets = (
                cat_inputs.to(device),
                num_inputs.to(device),
                targets.to(device),
            )

            preds = model(cat_inputs, num_inputs)

            loss = loss_fn(preds, targets)

            test_loss += loss.item()
            test_metric += metric_fn(preds, targets)

    test_loss /= len(test_dl)
    test_metric /= len(test_dl)

    return test_loss, test_metric


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
) -> None:
    best_val_loss = float("inf")
    patience_counter = 0
    temp_state_dict_path = Path("data/temp_best_state_dict.pt")

    for epoch in range(1, epochs + 1):
        train_loss, train_metric = train_step(
            model, train_dl, loss_fn, metric_fn, optimizer, device
        )
        val_loss, val_metric = test_step(model, val_dl, loss_fn, metric_fn, device)

        print(
            f"Epoch: {epoch} | Train loss: {train_loss:.2f} | Train {metric_fn.__name__}: {train_metric:.2f}% | Val loss: {val_loss:.2f} | Val {metric_fn.__name__}: {val_metric:.2f}%"
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
        model.load_state_dict(torch.load(temp_state_dict_path))
        print(f"Loaded best model parameters with val_loss={best_val_loss:.4f}")

    Path(temp_state_dict_path).unlink(missing_ok=True)


def test(
    model: nn.Module,
    data_loader: DataLoader,
    loss_fn: nn.Module,
    metric_fn: Callable,
    device: str = "cpu",
) -> None:
    test_loss, test_metric = test_step(model, data_loader, loss_fn, metric_fn, device)
    print(f"Test loss: {test_loss:.2f} | Test {metric_fn.__name__}: {test_metric:.2f}%")
