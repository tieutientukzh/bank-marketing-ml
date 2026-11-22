import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.data import RANDOM_STATE, load_dataset, split_train_test
from src.evaluate import eval_cv, eval_test, eval_top_k, find_best_threshold, plot_test_reports, run_grid
from src.features import DEFAULT_FEATURES, make_feature_step

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"


def build_prep():
    num_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("imp", SimpleImputer(missing_values="unknown", strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", num_pipe, make_column_selector(dtype_include="number")),
        ("cat", cat_pipe, make_column_selector(dtype_exclude="number")),
    ])


def make_pipe(clf, feat_names=DEFAULT_FEATURES):
    return Pipeline([
        ("feat", make_feature_step(feat_names)),
        ("prep", build_prep()),
        ("clf", clf),
    ])


def compare_features(X_train, y_train):
    subsets = {
        "4_No_Feat": (),
        "4_Core_Feat": ("contacted_before", "age_group", "drop_pdays"),
        "4_All_Feat": DEFAULT_FEATURES,
    }
    for label, names in subsets.items():
        pipe = make_pipe(LogisticRegression(C=5.0, max_iter=1000, random_state=RANDOM_STATE), names)
        sc, _ = eval_cv(pipe, X_train, y_train)
        print(f"{label}: PR-AUC = {sc:.4f}")


def print_top_features(pipe, top_n=8):
    names = pipe.named_steps["prep"].get_feature_names_out()
    coefs = pipe.named_steps["clf"].coef_[0]
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

    compare_features(X_train, y_train)

    log_grid = {"clf__C": [0.05, 0.1, 0.5, 1.0, 5.0], "clf__class_weight": [None, "balanced"]}
    best_log, log_params, log_sc = run_grid(make_pipe(LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)), log_grid, X_train, y_train)

    tree_grid = {"clf__max_depth": [4, 5, 6, 7, 8], "clf__min_samples_leaf": [10, 20, 50], "clf__class_weight": [None, "balanced"]}
    best_tree, tree_params, tree_sc = run_grid(make_pipe(DecisionTreeClassifier(random_state=RANDOM_STATE)), tree_grid, X_train, y_train)

    print(f"5_Best_Logistic: PR-AUC = {log_sc:.4f} | {log_params}")
    print(f"5_Best_Tree: PR-AUC = {tree_sc:.4f} | {tree_params}")

    best_pipe = best_log if log_sc >= tree_sc else best_tree
    best_thresh, cv_f1 = find_best_threshold(best_pipe, X_train, y_train)
    print(f"6_Threshold: {best_thresh:.4f} | CV_F1 = {cv_f1:.4f}")

    best_pipe.fit(X_train, y_train)
    probs = best_pipe.predict_proba(X_test)[:, 1]
    print(f"Test_Result: {eval_test(best_pipe, X_test, y_test, best_thresh)}")
    print(f"Top_20_Percent: {eval_top_k(y_test, probs, ratio=0.2)}")
    print("Top_Features:")
    print_top_features(best_pipe)

    plot_test_reports(y_test, probs, best_thresh)
    save_artifacts(best_pipe, best_thresh)


if __name__ == "__main__":
    main()