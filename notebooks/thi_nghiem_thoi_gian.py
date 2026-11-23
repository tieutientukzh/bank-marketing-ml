import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src.data import load_dataset, split_train_test
from src.pipeline import DATA_CLEANING, build_pipeline, make_model

ECONOMIC = ["emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed"]
TRAIN_RATIO = 0.8


def make_models():
    return {
        "Hoi quy logistic": make_model("logistic", C=1.0),
        "Cay quyet dinh": make_model("tree", max_depth=12, min_samples_leaf=100, class_weight="balanced"),
    }


def time_split(X, y):
    cut = int(len(X) * TRAIN_RATIO)
    return X.iloc[:cut], X.iloc[cut:], y.iloc[:cut], y.iloc[cut:]


def outside_train_range(X_train, X_test):
    low, high = X_train[ECONOMIC].min(), X_train[ECONOMIC].max()
    return ((X_test[ECONOMIC] < low) | (X_test[ECONOMIC] > high)).mean().round(3)


X, y = load_dataset()
splits = {"Ngau nhien": split_train_test(X, y), "Theo thoi gian": time_split(X, y)}

rows = []
for split_name, (X_train, X_test, y_train, y_test) in splits.items():
    for drop_economic in (False, True):
        columns = [c for c in X.columns if not (drop_economic and c in ECONOMIC)]
        for model_name, model in make_models().items():
            pipeline = build_pipeline(model, features=DATA_CLEANING).fit(X_train[columns], y_train)
            probs = pipeline.predict_proba(X_test[columns])[:, 1]
            rows.append({
                "cach_chia": split_name,
                "chi_so_kinh_te": "bo" if drop_economic else "giu",
                "mo_hinh": model_name,
                "ty_le_yes_train": y_train.mean(),
                "ty_le_yes_test": y_test.mean(),
                "pr_auc": average_precision_score(y_test, probs),
                "roc_auc": roc_auc_score(y_test, probs),
            })

result = pd.DataFrame(rows).round(4)
result.to_csv("reports/thi_nghiem_thoi_gian.csv", index=False)
print(result.to_string(index=False))
print()
X_train, X_test, _, _ = time_split(X, y)
print("Ti le dong test (theo thoi gian) nam ngoai khoang gia tri cua train:")
print(outside_train_range(X_train, X_test).to_string())
