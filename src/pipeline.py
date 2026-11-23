from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.data import RANDOM_STATE
from src.features import make_feature_step

NUMERIC_COLUMNS = make_column_selector(dtype_include="number")
CATEGORICAL_COLUMNS = make_column_selector(dtype_exclude="number")
DATA_CLEANING = ("contacted_before", "drop_pdays")
ALL_FEATURES = ("contacted_before", "age_group", "log_campaign", "drop_pdays")


def make_model(name, **params):
    if name == "dummy":
        return DummyClassifier(strategy="prior")
    if name == "logistic":
        return LogisticRegression(max_iter=5000, **params)
    if name == "tree":
        return DecisionTreeClassifier(random_state=RANDOM_STATE, **params)
    raise ValueError(f"Khong co mo hinh {name}")


def make_categorical_encoder(unknown_as_mode=False):
    encoder = OneHotEncoder(handle_unknown="ignore")
    if not unknown_as_mode:
        return encoder
    imputer = SimpleImputer(missing_values="unknown", strategy="most_frequent")
    return Pipeline([("dien_unknown", imputer), ("one_hot", encoder)])


def make_preprocessor(scale=True, unknown_as_mode=False):
    numeric_step = StandardScaler() if scale else "passthrough"
    return ColumnTransformer([
        ("so", numeric_step, NUMERIC_COLUMNS),
        ("phan_loai", make_categorical_encoder(unknown_as_mode), CATEGORICAL_COLUMNS),
    ])


def build_pipeline(model, features=(), scale=True, unknown_as_mode=False):
    steps = []
    if features:
        steps.append(("dac_trung", make_feature_step(features)))
    steps.append(("tien_xu_ly", make_preprocessor(scale, unknown_as_mode)))
    steps.append(("mo_hinh", model))
    return Pipeline(steps)