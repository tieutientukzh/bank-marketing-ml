import warnings

import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from src.data import load_dataset, split_train_test
from src.evaluate import evaluate_cv
from src.pipeline import ALL_FEATURES, DATA_CLEANING, build_pipeline, make_model

warnings.simplefilter("ignore", ConvergenceWarning)

TREE_DEPTH = 5
EXPERIMENTS = [
    ("0. Luon doan no", "dummy", {}, False),
    ("1. Du lieu tho", "raw", {"scale": False}, False),
    ("2. + Xu ly du lieu", "clean", {"features": DATA_CLEANING}, False),
    ("2b. Doi chung: unknown -> mode", "clean", {"features": DATA_CLEANING, "unknown_as_mode": True}, False),
    ("3. + Dac trung moi", "features", {"features": ALL_FEATURES}, False),
    ("Tham chieu: co duration", "clean", {"features": DATA_CLEANING}, True),
]


def models_for(kind):
    if kind == "dummy":
        return {"Moc so sanh": make_model("dummy")}
    return {
        "Hoi quy logistic": make_model("logistic"),
        "Cay quyet dinh": make_model("tree", max_depth=TREE_DEPTH),
    }


rows = []
for version, kind, options, keep_duration in EXPERIMENTS:
    X, y = load_dataset(keep_duration=keep_duration)
    X_train, X_test, y_train, y_test = split_train_test(X, y)
    for model_name, model in models_for(kind).items():
        scores = evaluate_cv(build_pipeline(model, **options), X_train, y_train)
        rows.append({"phien_ban": version, "mo_hinh": model_name, **scores})

result = pd.DataFrame(rows).round(4)
result.to_csv("reports/bang_thi_nghiem.csv", index=False)
print(result.to_string(index=False))
print()
print(result.pivot_table(index="phien_ban", columns="mo_hinh", values="pr_auc", sort=False).to_string())
