from sklearn.model_selection import StratifiedKFold, cross_validate
from src.data import RANDOM_STATE
CV = StratifiedKFold(n_splits = 5, shuffle=True, random_state = RANDOM_STATE)
SCORING = {"pr_auc": "average_precision", "roc_auc":"roc_auc"}
def count_iterations(fitted_pipelines):
    models = [pipeline[-1] for pipeline in fitted_pipelines]
    if not hasattr(models[0], "n_iter_"):
        return None
    return int(max(model.n_iter_.max() for model in models))
def evaluate_cv(pipeline, X, y):
    scores = cross_validate(pipeline, X, y, cv = CV, scoring =SCORING,
                            return_train_score = True, return_estimator=True)
    return {
        "pr_auc": scores["test_pr_auc"].mean(),
        "pr_auc_std": scores["test_pr_auc"].std(),
        "pr_auc_train": scores["train_pr_auc"].mean(),
        "roc_auc": scores["test_roc_auc"].mean(),
        "so_vong_lap": count_iterations(scores["estimator"]),
    }
