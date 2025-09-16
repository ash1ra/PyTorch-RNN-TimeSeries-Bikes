from pathlib import Path

import kagglehub
import numpy as np
import pandas as pd

path = Path(kagglehub.dataset_download("marklvl/bike-sharing-dataset"))

df = pd.read_csv(path / "day.csv")

df.rename(
    columns={
        "dteday": "date",
        "yr": "year",
        "mnth": "month",
        "cnt": "count",
    },
    inplace=True,
)

print(df.columns)

df["season"] = pd.Categorical(df["season"], categories=[1, 2, 3, 4], ordered=False)
df["year"] = pd.Categorical(df["year"], categories=[0, 1], ordered=False)
df["month"] = pd.Categorical(
    df["month"], categories=[i for i in range(1, 13)], ordered=False
)
df["holiday"] = pd.Categorical(df["holiday"], categories=[0, 1], ordered=False)
df["weekday"] = pd.Categorical(df["weekday"], categories=[i for i in range(7)], ordered=False)
df["workingday"] = pd.Categorical(df["workingday"], categories=[0, 1], ordered=False)
df["weathersit"] = pd.Categorical(df["weathersit"], categories=[1, 2, 3], ordered=False)

df["casual"] = np.log(df["casual"])
df["registered"] = np.log(df["registered"])
df["count"] = np.log(df["count"])

print(df.select_dtypes(include=["number"]).describe())
