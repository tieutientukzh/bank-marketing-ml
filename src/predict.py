import json
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
PIPELINE_PATH = MODELS_DIR / "pipeline.joblib"
THRESHOLD_PATH = MODELS_DIR / "threshold.json"
DEFAULT_THRESHOLD = 0.5

API_TO_COLUMN = {
    "emp_var_rate": "emp.var.rate",
    "cons_price_idx": "cons.price.idx",
    "cons_conf_idx": "cons.conf.idx",
    "nr_employed": "nr.employed",
}
INPUT_COLUMNS = [
    "age", "job", "marital", "education", "default", "housing", "loan",
    "contact", "month", "day_of_week", "campaign", "pdays", "previous", "poutcome",
    "emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed",
]


@lru_cache(maxsize=1)
def load_pipeline():
    return joblib.load(PIPELINE_PATH)


def load_threshold():
    if THRESHOLD_PATH.exists():
        return float(json.loads(THRESHOLD_PATH.read_text())["threshold"])
    return DEFAULT_THRESHOLD


def customer_to_frame(customer):
    renamed = {API_TO_COLUMN.get(key, key): value for key, value in customer.items()}
    missing = [column for column in INPUT_COLUMNS if column not in renamed]
    if missing:
        raise ValueError(f"Thieu thuoc tinh: {missing}")
    return pd.DataFrame([renamed], columns=INPUT_COLUMNS)


def predict_customer(customer, pipeline=None, threshold=None):
    pipeline = load_pipeline() if pipeline is None else pipeline
    threshold = load_threshold() if threshold is None else threshold
    probability = float(pipeline.predict_proba(customer_to_frame(customer))[0, 1])
    return {
        "probability": round(probability, 4),
        "prediction": "yes" if probability >= threshold else "no",
        "threshold": threshold,
    }
