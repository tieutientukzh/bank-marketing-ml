from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "bank-additional-full.csv"
TARGET = "y"
LEAK_COLUMN = "duration"
RANDOM_STATE = 42


def load_raw():
    return pd.read_csv(DATA_PATH, sep=";")


def load_dataset(keep_duration=False):
    df = load_raw().drop_duplicates().reset_index(drop=True)
    if not keep_duration:
        df = df.drop(columns=[LEAK_COLUMN])
    X = df.drop(columns=[TARGET])
    y = (df[TARGET] == "yes").astype(int)
    return X, y


def split_train_test(X, y, test_size=0.2):
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=RANDOM_STATE)