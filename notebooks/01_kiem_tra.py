import pandas as pd

df = pd.read_csv("data/raw/bank-additional-full.csv", sep=";")
print(df.shape)
print(df.dtypes)
print(df["y"].value_counts())
print(df["y"].value_counts(normalize=True).round(4))
print((df == "unknown").sum().loc[lambda s: s > 0])
print((df["pdays"] == 999).mean().round(4))
print(df.duplicated().sum())