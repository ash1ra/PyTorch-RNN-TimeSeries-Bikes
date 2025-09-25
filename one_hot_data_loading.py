from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np


TRAIN_RATIO = 0.7
VAL_RATIO = 0.15


class TimeSeriesDataset(Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
        target_col: str,
        feature_cols: list[str],
        seq_length: int,
    ) -> None:
        self.features = data[feature_cols].values.astype(np.float32)
        self.targets = data[target_col].values.astype(np.float32)
        self.seq_length = seq_length

    def __len__(self) -> int:
        return len(self.features) - self.seq_length

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.tensor(
                self.features[idx : idx + self.seq_length], dtype=torch.float32
            ),
            torch.tensor(self.targets[idx + self.seq_length], dtype=torch.float32),
        )


def split_data(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    total_size = len(data)
    train_size = int(TRAIN_RATIO * total_size)
    val_size = int(VAL_RATIO * total_size)

    return (
        data.iloc[:train_size],
        data.iloc[train_size : train_size + val_size],
        data.iloc[train_size + val_size :],
    )


def save_datasets(
    output_dir: str,
    train_ds: Dataset,
    val_ds: Dataset,
    test_ds: Dataset,
) -> None:
    output_path = Path(output_dir)
    torch.save(train_ds, output_path / "train_ds.pt")
    torch.save(val_ds, output_path / "val_ds.pt")
    torch.save(test_ds, output_path / "test_ds.pt")


def get_dataloaders(
    train_ds_path, val_ds_path, test_ds_path, batch_size
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_ds = torch.load(train_ds_path, weights_only=False)
    val_ds = torch.load(val_ds_path, weights_only=False)
    test_ds = torch.load(test_ds_path, weights_only=False)

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=False),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False),
    )


path = Path("data")
df = pd.read_csv(path / "processed_data.csv")

train_df, val_df, test_df = split_data(df)

target_col = "count"
feature_cols = [col for col in df.columns if col != target_col]

train_ds = TimeSeriesDataset(train_df, target_col, feature_cols, 30)
val_ds = TimeSeriesDataset(val_df, target_col, feature_cols, 30)
test_ds = TimeSeriesDataset(test_df, target_col, feature_cols, 30)

save_datasets("data", train_ds, val_ds, test_ds)
