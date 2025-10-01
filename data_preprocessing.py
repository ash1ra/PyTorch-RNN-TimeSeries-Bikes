from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pandas.plotting import autocorrelation_plot
from sklearn.preprocessing import StandardScaler

path = Path("data")
df = pd.read_csv(path / "hour.csv")

df.rename(
    columns={
        "dteday": "date",
        "yr": "year",
        "mnth": "month",
        "hr": "hour",
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

df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

df["day_of_week_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
df["day_of_week_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

df["season"] = pd.Categorical(df["season"], categories=[1, 2, 3, 4], ordered=False)
df["year"] = pd.Categorical(df["year"], categories=[0, 1], ordered=False)
df["is_holiday"] = pd.Categorical(df["is_holiday"], categories=[0, 1], ordered=False)
df["is_working_day"] = pd.Categorical(
    df["is_working_day"], categories=[0, 1], ordered=False
)
df["weather"] = pd.Categorical(df["weather"], categories=[1, 2, 3, 4], ordered=False)

df["season"] = df["season"].cat.codes
df["year"] = df["year"].cat.codes
df["is_holiday"] = df["is_holiday"].cat.codes
df["is_working_day"] = df["is_working_day"].cat.codes
df["weather"] = df["weather"].cat.codes

df["count"] = np.log(df["count"])

df["count_lag_1"] = df["count"].shift(1)
df["count_lag_24"] = df["count"].shift(24)
df["count_lag_168"] = df["count"].shift(168)

df = df.dropna()

cols_to_scale = [
    "temp",
    "hum",
    "windspeed",
    "count",
    "count_lag_1",
    "count_lag_24",
    "count_lag_168",
]

scaler = StandardScaler()
df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])

count_index = cols_to_scale.index("count")
COUNT_MEAN = scaler.mean_[count_index]
COUNT_STD = scaler.scale_[count_index]

columns_to_export = [
    "year",
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
    "day_of_week_sin",
    "day_of_week_cos",
    "is_holiday",
    "is_working_day",
    "season",
    "weather",
    "feeling_temp",
    "hum",
    "windspeed",
    "count_lag_1",
    "count_lag_24",
    "count_lag_168",
    "count",
]

# autocorrelation_plot(df["count"][:50])
# plt.show()

# corr_matrix = df[columns_to_export].corr(method="pearson")
# sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
# plt.show()

df[columns_to_export].to_csv(path / "processed_data.csv", index=False)
