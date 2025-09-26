from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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

df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
df.drop(columns=["month"], inplace=True)

df["day_of_week_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
df["day_of_week_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
df.drop(columns=["day_of_week"], inplace=True)

cat_cols = [
    "season",
    "year",
    "is_holiday",
    "is_working_day",
    "weather",
]

num_cols = [
    "month_sin",
    "month_cos",
    "day_of_week_sin",
    "day_of_week_cos",
    "temp",
    "feeling_temp",
    "hum",
    "windspeed",
]

df = pd.get_dummies(df, columns=cat_cols, dtype=int)

df.loc[df["date"] == pd.to_datetime("2012-10-29"), "count"] = (
    df["count"]
    .rolling(window=2, center=True)
    .mean()
    .loc[df["date"] == pd.to_datetime("2012-10-29")]
    .astype(int)
)

df["count"] = np.log(df["count"])

df.drop(columns=["instant", "date", "casual", "registered"], inplace=True)


columns_to_export = [col for col in df.columns]

df[columns_to_export].to_csv(path / "processed_data.csv", index=False)
