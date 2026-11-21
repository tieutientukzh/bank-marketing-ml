import numpy as np
import pandas as pd
from sklearn.preprocessing import FunctionTransformer

NEVER_CONTACTED_CODE = 999
AGE_BIN_EDGES = [0, 24, 60, 200]
AGE_GROUP_LABELS = ["under_25", "25_to_60", "over_60"]


def add_contacted_before(df):
    out = df.copy()
    out["contacted_before"] = (out["pdays"] != NEVER_CONTACTED_CODE).astype(int)
    return out


def add_age_group(df):
    out = df.copy()
    out["age_group"] = pd.cut(out["age"], bins=AGE_BIN_EDGES, labels=AGE_GROUP_LABELS).astype(str)
    return out


def log_campaign(df):
    out = df.copy()
    out["campaign"] = np.log1p(out["campaign"])
    return out


def drop_pdays(df):
    return df.drop(columns=["pdays"])


FEATURE_FUNCTIONS = {
    "contacted_before": add_contacted_before,
    "age_group": add_age_group,
    "log_campaign": log_campaign,
    "drop_pdays": drop_pdays,
}
DEFAULT_FEATURES = tuple(FEATURE_FUNCTIONS)


def add_features(df, names=DEFAULT_FEATURES):
    for name in names:
        df = FEATURE_FUNCTIONS[name](df)
    return df


def make_feature_step(names=DEFAULT_FEATURES):
    return FunctionTransformer(add_features, kw_args={"names": tuple(names)})
