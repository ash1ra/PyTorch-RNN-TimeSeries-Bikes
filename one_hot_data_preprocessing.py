from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


path = Path("data")
df = pd.read_csv(path / "data.csv")

df.rename(
    columns={
        "dteday": "date",
        "yr": "year",
        "mnth": "month",
        "holiday": "is_holiday",
        "weekday": "day_of_week",
        "workingday": "is_working_day",
        "weathersit": "weather",
        "atemp": "feeling_temp",
        "cnt": "count",
    },
    inplace=True,
)

df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")
df = df.sort_values("date")

# df["day"] = df["date"].dt.day

cat_cols = [
    "season",
    "year",
    "month",
    "is_holiday",
    "day_of_week",
    "is_working_day",
    "weather",
]
num_cols = ["temp", "feeling_temp", "hum", "windspeed"]

df = pd.get_dummies(df, columns=cat_cols, dtype=int)

df.loc[df["date"] == pd.to_datetime("2012-10-29"), "count"] = (
    df["count"]
    .rolling(window=2, center=True)
    .mean()
    .loc[df["date"] == pd.to_datetime("2012-10-29")]
    .astype(int)
)

df["count"] = np.log(df["count"])

columns_to_export = [
    col for col in df.columns if col not in ["instant", "date", "casual", "registered"]
]

df[columns_to_export].to_csv(path / "processed_data.csv", index=False)
