from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.data import load_raw

FIGURES = Path("reports/figures")
FIGURES.mkdir(parents=True, exist_ok=True)
CATEGORICAL = ["job", "marital", "education", "default", "housing", "loan",
               "contact", "month", "day_of_week", "poutcome"]
ECONOMIC = ["emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed"]
NUMERIC = ["age", "campaign", "pdays", "previous"] + ECONOMIC

df = load_raw().drop_duplicates().reset_index(drop=True)
df["subscribed"] = (df["y"] == "yes").astype(int)

print(df["default"].value_counts())
print(((df["housing"] == "unknown") == (df["loan"] == "unknown")).all())
print(pd.crosstab(df["pdays"] == 999, df["poutcome"]))
print(df[NUMERIC].describe().T.round(2))
print(df[ECONOMIC].nunique())

for column in CATEGORICAL:
    rate = df.groupby(column)["subscribed"].agg(["mean", "count"]).sort_values("mean", ascending=False)
    print(rate.round(3))

fig, axes = plt.subplots(2, 5, figsize=(24, 9))
for ax, column in zip(axes.ravel(), CATEGORICAL):
    df.groupby(column)["subscribed"].mean().sort_values().plot.barh(ax=ax, color="#2F4B8F")
    ax.axvline(df["subscribed"].mean(), color="#F28C28", linestyle="--")
    ax.set_title(column)
    ax.set_xlabel("tỉ lệ yes")
fig.tight_layout()
fig.savefig(FIGURES / "ti_le_yes_theo_nhom.png", dpi=150)

print(df.groupby("y")["duration"].describe().round(1))
print(df.loc[df["duration"] == 0, "y"].value_counts())
fig, ax = plt.subplots(figsize=(8, 5))
sns.boxplot(data=df, x="y", y="duration", showfliers=False, ax=ax)
ax.set_title("Thời lượng cuộc gọi theo kết quả")
fig.savefig(FIGURES / "duration_theo_y.png", dpi=150)

fig, ax = plt.subplots(figsize=(9, 5))
sns.histplot(data=df, x="age", hue="y", stat="density", common_norm=False, bins=40, ax=ax)
ax.set_title("Phân phối tuổi theo kết quả")
fig.savefig(FIGURES / "tuoi_theo_y.png", dpi=150)

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(df[NUMERIC + ["subscribed"]].corr(), annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax)
ax.set_title("Tương quan giữa các biến số")
fig.tight_layout()
fig.savefig(FIGURES / "tuong_quan.png", dpi=150)
