import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.data import load_dataset, split_train_test
from src.evaluate import (
    analyze_errors,
    eval_cv,
    eval_test,
    eval_top_k,
    find_best_threshold,
    plot_test_reports,
    run_grid,
)
from src.pipeline import ALL_FEATURES, DATA_CLEANING, build_pipeline, make_model

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
RESULTS_PATH = ROOT / "reports" / "ket_qua_mo_hinh.csv"
FEATURE_SUBSETS = {
    "4. Chon dac trung: chi xu ly du lieu": DATA_CLEANING,
    "4. Chon dac trung: bo log_campaign": ("contacted_before", "age_group", "drop_pdays"),
    "4. Chon dac trung: day du": ALL_FEATURES,
}
LOGISTIC_GRID = {
    "mo_hinh__C": [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0],
    "mo_hinh__class_weight": [None, "balanced"],
}
TREE_GRID = {
    "mo_hinh__max_depth": [4, 6, 8, 10, 12],
    "mo_hinh__min_samples_leaf": [10, 20, 50, 100],
    "mo_hinh__class_weight": [None, "balanced"],
}


def compare_features(X_train, y_train):
    rows = []
    for label, names in FEATURE_SUBSETS.items():
        pipe = build_pipeline(make_model("logistic", C=5.0), features=names)
        score, std = eval_cv(pipe, X_train, y_train)
        print(f"{label}: PR-AUC = {score:.4f}")
        rows.append({"phien_ban": label, "mo_hinh": "Hoi quy logistic", "pr_auc": score, "pr_auc_std": std})
    best = max(rows, key=lambda row: row["pr_auc"])
    print(f"Chon: {best['phien_ban']}")
    return rows, FEATURE_SUBSETS[best["phien_ban"]]


def print_top_features(pipe, top_n=8):
    names = pipe.named_steps["tien_xu_ly"].get_feature_names_out()
    coefs = pipe.named_steps["mo_hinh"].coef_[0]
    order = np.argsort(np.abs(coefs))[::-1][:top_n]
    for idx in order:
        print(f"  {names[idx]}: {coefs[idx]:+.4f}")


def save_artifacts(pipe, threshold):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, MODELS_DIR / "pipeline.joblib")
    (MODELS_DIR / "threshold.json").write_text(json.dumps({"threshold": round(threshold, 4)}, indent=2))


def main():
    X, y = load_dataset(keep_duration=False)
    X_train, X_test, y_train, y_test = split_train_test(X, y)

    rows, features = compare_features(X_train, y_train)

    log_pipe = build_pipeline(make_model("logistic"), features=features)
    best_log, log_params, log_sc = run_grid(log_pipe, LOGISTIC_GRID, X_train, y_train)

    tree_pipe = build_pipeline(make_model("tree"), features=features)
    best_tree, tree_params, tree_sc = run_grid(tree_pipe, TREE_GRID, X_train, y_train)

    print(f"5-6. Best_Logistic: PR-AUC = {log_sc:.4f} | {log_params}")
    print(f"5-6. Best_Tree: PR-AUC = {tree_sc:.4f} | {tree_params}")
    for name, score, params in (("Hoi quy logistic", log_sc, log_params), ("Cay quyet dinh", tree_sc, tree_params)):
        rows.append({
            "phien_ban": "5-6. Tinh chinh + class_weight",
            "mo_hinh": name,
            "pr_auc": score,
            "tham_so": str(params),
        })

    best_pipe = best_log if log_sc >= tree_sc else best_tree
    best_thresh, cv_f1 = find_best_threshold(best_pipe, X_train, y_train)
    print(f"6_Threshold: {best_thresh:.4f} | CV_F1 = {cv_f1:.4f}")

    best_pipe.fit(X_train, y_train)
    probs = best_pipe.predict_proba(X_test)[:, 1]
    test_result = eval_test(best_pipe, X_test, y_test, best_thresh)
    top_k = eval_top_k(y_test, probs, ratio=0.2)
    print(f"Test_Result: {test_result}")
    print(f"Top_20_Percent: {top_k}")
    rows.append({
        "phien_ban": "Ket qua tren tap test",
        "mo_hinh": "Mo hinh cuoi",
        "pr_auc": test_result["pr_auc"],
        "nguong": best_thresh,
        "f1": test_result["f1"],
        "precision": test_result["precision"],
        "recall": test_result["recall"],
        "recall_goi_20": top_k["recall_at_20"],
    })
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).round(4).to_csv(RESULTS_PATH, index=False)

    if hasattr(best_pipe.named_steps["mo_hinh"], "coef_"):
        print("Top_Features:")
        print_top_features(best_pipe)

    errors = analyze_errors(X_test, y_test, probs, best_thresh)
    print(f"Errors: FP = {errors['fp_count']}, FN = {errors['fn_count']}")
    print("Top_FP (Doan nham yes):")
    print(errors["fp_top"].to_string(index=False))
    print("Top_FN (Bo sot yes):")
    print(errors["fn_top"].to_string(index=False))

    plot_test_reports(y_test, probs, best_thresh)
    save_artifacts(best_pipe, best_thresh)


if __name__ == "__main__":
    main()
