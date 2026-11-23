import pandas as pd

from src.data import load_dataset, split_train_test
from src.evaluate import evaluate_cv
from src.pipeline import build_pipeline, make_model

MODELS = {
    "Hoi quy logistic": make_model("logistic"),
    "Cay quyet dinh": make_model("tree", max_depth=5),
}

rows = []
for keep_duration in (False, True):
    X, y = load_dataset(keep_duration=keep_duration)
    X_train, X_test, y_train, y_test = split_train_test(X, y)
    for name, model in MODELS.items():
        scores = evaluate_cv(build_pipeline(model), X_train, y_train)
        rows.append({"mo_hinh": name, "duration": "co" if keep_duration else "khong", **scores})

result = pd.DataFrame(rows).sort_values(["mo_hinh", "duration"]).round(4)
print(f"PR-AUC khi doan ngau nhien = {y_train.mean():.4f}")
print(result.to_string(index=False))
result.to_csv("reports/thi_nghiem_duration.csv", index=False)
