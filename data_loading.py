from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, load_from_disk
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15

SEQ_LENGTH = 96
TARGET_COL = "count"
CAT_COLS = [
    "year",
    "is_holiday",
    "is_working_day",
    "season",
    "weather",
]
NUM_COLS = [
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
    "day_of_week_sin",
    "day_of_week_cos",
    "feeling_temp",
    "hum",
    "windspeed",
    "count_lag_1",
    "count_lag_24",
    "count_lag_168",
]


class TimeSeriesDataset(TorchDataset):
    def __init__(
        self,
        data: pd.DataFrame,
        target_col: str,
        cat_cols: list[str],
        num_cols: list[str],
        seq_length: int,
    ) -> None:
        self.cat_data = data[cat_cols].values.astype(np.int64)
        self.num_data = data[num_cols].values.astype(np.float32)
        self.targets = data[target_col].values.astype(np.float32)
        self.seq_length = seq_length
        self.cat_cols = cat_cols
        self.num_cols = num_cols
        self.target_col = target_col

    def __len__(self) -> int:
        return len(self.num_data) - self.seq_length

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.tensor(self.cat_data[idx : idx + self.seq_length], dtype=torch.int64),
            torch.tensor(
                self.num_data[idx : idx + self.seq_length], dtype=torch.float32
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


def create_dataframe(ds: TimeSeriesDataset) -> pd.DataFrame:
    cat_dict = {col: ds.cat_data[:, i] for i, col in enumerate(ds.cat_cols)}
    num_dict = {col: ds.num_data[:, i] for i, col in enumerate(ds.num_cols)}
    data_dict = {**cat_dict, **num_dict, ds.target_col: ds.targets}

    return pd.DataFrame(data_dict)


def save_datasets(
    output_dir: str,
    train_ds: TimeSeriesDataset,
    val_ds: TimeSeriesDataset,
    test_ds: TimeSeriesDataset,
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    train_df = create_dataframe(train_ds)
    val_df = create_dataframe(val_ds)
    test_df = create_dataframe(test_ds)

    Dataset.from_pandas(train_df).save_to_disk(output_path / "train_dataset")
    Dataset.from_pandas(val_df).save_to_disk(output_path / "val_dataset")
    Dataset.from_pandas(test_df).save_to_disk(output_path / "test_dataset")


def get_dataloaders(
    train_ds_path: Path,
    val_ds_path: Path,
    test_ds_path: Path,
    batch_size: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_df = load_from_disk(train_ds_path).to_pandas()
    val_df = load_from_disk(val_ds_path).to_pandas()
    test_df = load_from_disk(test_ds_path).to_pandas()

    train_ds = TimeSeriesDataset(train_df, TARGET_COL, CAT_COLS, NUM_COLS, SEQ_LENGTH)
    val_ds = TimeSeriesDataset(val_df, TARGET_COL, CAT_COLS, NUM_COLS, SEQ_LENGTH)
    test_ds = TimeSeriesDataset(test_df, TARGET_COL, CAT_COLS, NUM_COLS, SEQ_LENGTH)

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=False, num_workers=2),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2),
    )


if __name__ == "__main__":
    path = Path("data")
    df = pd.read_csv(path / "processed_data.csv")

    train_df, val_df, test_df = split_data(df)

    train_ds = TimeSeriesDataset(train_df, TARGET_COL, CAT_COLS, NUM_COLS, SEQ_LENGTH)
    val_ds = TimeSeriesDataset(val_df, TARGET_COL, CAT_COLS, NUM_COLS, SEQ_LENGTH)
    test_ds = TimeSeriesDataset(test_df, TARGET_COL, CAT_COLS, NUM_COLS, SEQ_LENGTH)

    save_datasets("data", train_ds, val_ds, test_ds)
