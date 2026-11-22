from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict, cross_val_score, learning_curve, validation_curve

FIG_DIR = Path(__file__).resolve().parents[1] / "reports" / "figures"


def get_cv(n_splits=5):
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)


def eval_cv(model, X, y):
    scores = cross_val_score(model, X, y, cv=get_cv(), scoring="average_precision", n_jobs=-1)
    return float(scores.mean()), float(scores.std())


def run_grid(pipe, grid, X, y):
    search = GridSearchCV(pipe, grid, cv=get_cv(), scoring="average_precision", n_jobs=-1)
    search.fit(X, y)
    return search.best_estimator_, search.best_params_, float(search.best_score_)


def find_best_threshold(pipe, X, y):
    probs = cross_val_predict(pipe, X, y, cv=get_cv(), method="predict_proba", n_jobs=-1)[:, 1]
    prec, rec, thresh = precision_recall_curve(y, probs)
    f1_scores = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-12)
    best_idx = int(np.argmax(f1_scores))
    return float(thresh[best_idx]), float(f1_scores[best_idx])


def eval_test(pipe, X_test, y_test, threshold):
    probs = pipe.predict_proba(X_test)[:, 1]
    preds = (probs >= threshold).astype(int)
    return {
        "pr_auc": float(average_precision_score(y_test, probs)),
        "f1": float(f1_score(y_test, preds)),
        "precision": float(precision_score(y_test, preds)),
        "recall": float(recall_score(y_test, preds)),
    }


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