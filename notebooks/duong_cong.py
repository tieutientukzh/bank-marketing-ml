import numpy as np

from src.data import load_dataset, split_train_test
from src.evaluate import plot_learn_curve, plot_val_curve
from src.pipeline import ALL_FEATURES, build_pipeline, make_model


def main():
    X, y = load_dataset(keep_duration=False)
    X_train, X_test, y_train, y_test = split_train_test(X, y)
    logistic = build_pipeline(make_model("logistic"), features=ALL_FEATURES)
    tree = build_pipeline(make_model("tree"), features=ALL_FEATURES)

    plot_val_curve(
        logistic, X_train, y_train, "mo_hinh__C", np.logspace(-3, 2, 6),
        "Validation curve: hoi quy logistic theo C", "val_logistic_C.png", is_log=True,
    )
    plot_val_curve(
        tree, X_train, y_train, "mo_hinh__max_depth", [2, 4, 6, 8, 10, 12, 15, 20],
        "Validation curve: cay quyet dinh theo max_depth", "val_tree_depth.png",
    )
    plot_learn_curve(logistic, X_train, y_train, "Learning curve: hoi quy logistic", "learn_logistic.png")
    plot_learn_curve(
        tree.set_params(mo_hinh__max_depth=8), X_train, y_train,
        "Learning curve: cay quyet dinh (max_depth = 8)", "learn_tree.png",
    )
    print("Da luu 4 hinh vao reports/figures")


if __name__ == "__main__":
    main()
