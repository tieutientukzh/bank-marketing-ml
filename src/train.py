import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.data import RANDOM_STATE, load_dataset, split_train_test
from src.evaluate import eval_cv, plot_learn_curve, plot_val_curve


def build_prep(X):
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X.select_dtypes(exclude=["object", "category"]).columns.tolist()
    return ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ])


def get_models(X):
    return {
        "0_Dummy": Pipeline([("prep", build_prep(X)), ("clf", DummyClassifier(strategy="most_frequent"))]),
        "1_Logistic": Pipeline([("prep", build_prep(X)), ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))]),
        "1_Tree": Pipeline([("prep", build_prep(X)), ("clf", DecisionTreeClassifier(random_state=RANDOM_STATE))]),
    }


def run_baselines(X_train, y_train):
    for name, model in get_models(X_train).items():
        mean_sc, std_sc = eval_cv(model, X_train, y_train)
        print(f"{name}: PR-AUC = {mean_sc:.4f} (+/- {std_sc:.4f})")


def run_curves(X_train, y_train):
    models = get_models(X_train)
    plot_val_curve(models["1_Logistic"], X_train, y_train, "clf__C", np.logspace(-3, 2, 6), "Logistic C", "val_logistic_C.png", True)
    plot_val_curve(models["1_Tree"], X_train, y_train, "clf__max_depth", [2, 4, 6, 8, 10, 15, 20], "Tree Depth", "val_tree_depth.png")
    plot_learn_curve(models["1_Logistic"], X_train, y_train, "Learning Logistic", "learn_logistic.png")
    plot_learn_curve(models["1_Tree"], X_train, y_train, "Learning Tree", "learn_tree.png")


def main():
    X, y = load_dataset(keep_duration=False)
    X_train, _, y_train, _ = split_train_test(X, y)
    run_baselines(X_train, y_train)
    run_curves(X_train, y_train)


if __name__ == "__main__":
    main()