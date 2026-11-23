from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
    cross_validate,
    learning_curve,
    validation_curve,
)

from src.data import RANDOM_STATE

FIG_DIR = Path(__file__).resolve().parents[1] / "reports" / "figures"
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
SCORING = {"pr_auc": "average_precision", "roc_auc": "roc_auc"}


def count_iterations(fitted_pipelines):
    models = [pipeline[-1] for pipeline in fitted_pipelines]
    if not hasattr(models[0], "n_iter_"):
        return None
    return int(max(model.n_iter_.max() for model in models))


def evaluate_cv(pipeline, X, y):
    scores = cross_validate(
        pipeline, X, y, cv=CV, scoring=SCORING, return_train_score=True, return_estimator=True
    )
    return {
        "pr_auc": scores["test_pr_auc"].mean(),
        "pr_auc_std": scores["test_pr_auc"].std(),
        "pr_auc_train": scores["train_pr_auc"].mean(),
        "roc_auc": scores["test_roc_auc"].mean(),
        "so_vong_lap": count_iterations(scores["estimator"]),
    }


def get_cv(n_splits=5):
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)


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


def eval_top_k(y_test, probs, ratio=0.2):
    k = int(len(y_test) * ratio)
    top_idx = np.argsort(probs)[::-1][:k]
    y_arr = np.asarray(y_test)
    hits = int(y_arr[top_idx].sum())
    total_yes = int(y_arr.sum())
    return {
        "called": k,
        "hits": hits,
        "precision_at_20": round(hits / k, 4),
        "recall_at_20": round(hits / total_yes, 4),
    }


def analyze_errors(X_test, y_test, probs, threshold, top_n=5):
    df = X_test.copy()
    df["y_true"] = np.asarray(y_test)
    df["prob"] = probs
    df["y_pred"] = (probs >= threshold).astype(int)

    fp = df[(df["y_true"] == 0) & (df["y_pred"] == 1)].sort_values("prob", ascending=False)
    fn = df[(df["y_true"] == 1) & (df["y_pred"] == 0)].sort_values("prob", ascending=True)
    cols = ["age", "job", "contact", "month", "pdays", "poutcome", "emp.var.rate", "prob"]

    return {
        "fp_count": len(fp),
        "fn_count": len(fn),
        "fp_top": fp[cols].head(top_n),
        "fn_top": fn[cols].head(top_n),
    }


def plot_test_reports(y_test, probs, threshold):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    preds = (probs >= threshold).astype(int)

    cm = confusion_matrix(y_test, preds)
    ConfusionMatrixDisplay(cm, display_labels=["no", "yes"]).plot(cmap="Blues")
    plt.title(f"Ma tran nham lan (nguong = {threshold:.4f})")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "confusion_matrix.png", dpi=150)
    plt.close()

    PrecisionRecallDisplay.from_predictions(y_test, probs)
    plt.title("Duong cong Precision-Recall (Tap Test)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "pr_curve.png", dpi=150)
    plt.close()


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