from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.data import RANDOM_STATE, load_dataset, split_train_test
from src.evaluate import eval_cv, run_grid
from src.features import make_feature_step


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


def make_pipe(clf):
    return Pipeline([
        ("feat", make_feature_step()),
        ("prep", build_prep()),
        ("clf", clf),
    ])


def run_n2_morning(X_train, y_train):
    dummy = make_pipe(DummyClassifier(strategy="most_frequent"))
    d_mean, _ = eval_cv(dummy, X_train, y_train)
    print(f"0_Dummy: PR-AUC = {d_mean:.4f}")

    log_pipe = make_pipe(LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))
    log_grid = {
        "clf__C": [0.05, 0.1, 0.5, 1.0, 5.0],
        "clf__class_weight": [None, "balanced"],
    }
    _, log_params, log_sc = run_grid(log_pipe, log_grid, X_train, y_train)
    print(f"Logistic Grid: PR-AUC = {log_sc:.4f} | {log_params}")

    tree_pipe = make_pipe(DecisionTreeClassifier(random_state=RANDOM_STATE))
    tree_grid = {
        "clf__max_depth": [4, 5, 6, 7, 8],
        "clf__min_samples_leaf": [10, 20, 50],
        "clf__class_weight": [None, "balanced"],
    }
    _, tree_params, tree_sc = run_grid(tree_pipe, tree_grid, X_train, y_train)
    print(f"Tree Grid: PR-AUC = {tree_sc:.4f} | {tree_params}")


def main():
    X, y = load_dataset(keep_duration=False)
    X_train, _, y_train, _ = split_train_test(X, y)
    run_n2_morning(X_train, y_train)


if __name__ == "__main__":
    main()