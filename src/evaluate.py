from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score, learning_curve, validation_curve

FIG_DIR = Path(__file__).resolve().parents[1] / "reports" / "figures"


def get_cv(n_splits=5):
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)


def eval_cv(model, X, y):
    scores = cross_val_score(model, X, y, cv=get_cv(), scoring="average_precision", n_jobs=-1)
    return float(scores.mean()), float(scores.std())


def save_plot(x_vals, train_sc, val_sc, xlabel, title, filename, is_log=False):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4))
    plt.plot(x_vals, train_sc.mean(axis=1), marker="o", label="Train")
    plt.plot(x_vals, val_sc.mean(axis=1), marker="s", label="Validation")
    if is_log:
        plt.xscale("log")
    plt.xlabel(xlabel)
    plt.ylabel("PR-AUC")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / filename, dpi=150)
    plt.close()


def plot_val_curve(model, X, y, param, param_range, title, filename, is_log=False):
    train_sc, val_sc = validation_curve(
        model, X, y, param_name=param, param_range=param_range, cv=get_cv(), scoring="average_precision", n_jobs=-1
    )
    save_plot(param_range, train_sc, val_sc, param, title, filename, is_log)


def plot_learn_curve(model, X, y, title, filename):
    sizes, train_sc, val_sc = learning_curve(
        model, X, y, train_sizes=np.linspace(0.1, 1.0, 6), cv=get_cv(), scoring="average_precision", n_jobs=-1
    )
    save_plot(sizes, train_sc, val_sc, "So mau huan luyen", title, filename)