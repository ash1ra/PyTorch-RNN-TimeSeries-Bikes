from pathlib import Path
from data_loading import get_dataloaders


def main() -> None:
    path = Path("data")

    train_ds_path = path / "train_ds.pt"
    val_ds_path = path / "val_ds.pt"
    test_ds_path = path / "test_ds.pt"

    train_dl, val_dl, test_dl = get_dataloaders(
        train_ds_path, val_ds_path, test_ds_path, 32
    )

    print(train_dl, val_dl, test_dl)
    for X, y in val_dl:
        print(X, y)


if __name__ == "__main__":
    main()
