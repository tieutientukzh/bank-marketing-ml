import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from src.data import RANDOM_STATE, load_dataset, split_train_test
CV = StratifiedKFold(n_splits = 5, shuffle = True, random_state = RANDOM_STATE)
SCORING = {"pr_auc": "average_precision", "roc_auc": "roc_auc"}
MODELS = {
    "hoi quy logistic": LogisticRegression(max_iter = 2000),
    "Cay quyet dinh": DecisionTreeClassifier(max_depth = 5, random_state =RANDOM_STATE),
}
def build_pipeline(model):
    preprocess = ColumnTransformer([
        ("So", StandardScaler(), make_column_selector(dtype_include="number")),
        ("phan_loai", OneHotEncoder(handle_unknown="ignore"), make_column_selector(dtype_exclude="number")),
    ])
    return Pipeline([("tien_xu_ly", preprocess), ("mo_hinh", model)])
rows = []
for keep_duration in(False, True):
    X, y = load_dataset(keep_duration= keep_duration)
    X_train, X_test, y_train, y_test = split_train_test(X, y)
    for name, model in MODELS.items():
        scores = cross_validate(build_pipeline(model), X_train, y_train, cv =CV, scoring = SCORING)
        rows.append({
            "mo_hinh":name,
            "duration":"co" if keep_duration else"khong",
            "pr_auc":scores["test_pr_auc"].mean(),
            "pr_auc_std":scores["test_pr_auc"].std(),
            "roc_auc":scores["test_roc_auc"].mean(),
        })
result = pd.DataFrame(rows).sort_values(["mo_hinh", "duration"]).round(4)
print(f"PR-AUC khi doan ngau nhien = {y_train.mean():.4f}")
print(result.to_string(index =False))
result.to_csv("reports/thi_nghiem_duration.csv", index=False)