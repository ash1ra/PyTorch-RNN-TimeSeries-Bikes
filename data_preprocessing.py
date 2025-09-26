from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

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

df["day_of_week_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
df["day_of_week_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

df["season"] = pd.Categorical(df["season"], categories=[1, 2, 3, 4], ordered=False)
df["year"] = pd.Categorical(df["year"], categories=[0, 1], ordered=False)
# df["month"] = pd.Categorical(
#     df["month"], categories=[i for i in range(1, 13)], ordered=False
# )
df["is_holiday"] = pd.Categorical(df["is_holiday"], categories=[0, 1], ordered=False)
# df["day_of_week"] = pd.Categorical(
#     df["day_of_week"], categories=[i for i in range(7)], ordered=False
# )
df["is_working_day"] = pd.Categorical(
    df["is_working_day"], categories=[0, 1], ordered=False
)
df["weather"] = pd.Categorical(df["weather"], categories=[1, 2, 3], ordered=False)

df["season"] = df["season"].cat.codes
df["year"] = df["year"].cat.codes
# df["month"] = df["month"].cat.codes
df["is_holiday"] = df["is_holiday"].cat.codes
# df["day_of_week"] = df["day_of_week"].cat.codes
df["is_working_day"] = df["is_working_day"].cat.codes
df["weather"] = df["weather"].cat.codes

df.loc[df["date"] == pd.to_datetime("2012-10-29"), "count"] = (
    df["count"]
    .rolling(window=2, center=True)
    .mean()
    .loc[df["date"] == pd.to_datetime("2012-10-29")]
    .astype(int)
)

num_cols = [
    "temp",
    "feeling_temp",
    "hum",
    "windspeed",
    "month_sin",
    "month_cos",
    "day_of_week_sin",
    "day_of_week_cos",
    # "count",
]

print(df[num_cols])
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])
print(df[num_cols])

df["count"] = np.log(df["count"])

columns_to_export = [
    "year",
    # "month",
    "month_sin",
    "month_cos",
    # "day_of_week",
    "day_of_week_sin",
    "day_of_week_cos",
    "is_holiday",
    "is_working_day",
    "season",
    "weather",
    "temp",
    "feeling_temp",
    "hum",
    "windspeed",
    "count",
]

df[columns_to_export].to_csv(path / "processed_data.csv", index=False)
