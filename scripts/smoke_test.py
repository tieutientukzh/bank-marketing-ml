from pydantic import ValidationError

from src.predict import predict_customer
from src.schema import SAMPLE_CUSTOMERS, Customer, category_options


def check_model_on_samples():
    results = {name: predict_customer(customer) for name, customer in SAMPLE_CUSTOMERS.items()}
    for name, result in results.items():
        print(f"{name:12s} xac suat = {result['probability']:.3f} -> {result['prediction']}")
    assert results["high_chance"]["probability"] > results["typical"]["probability"]
    assert results["typical"]["probability"] > results["low_chance"]["probability"]


def check_schema_rejects_bad_input():
    bad_age = {**SAMPLE_CUSTOMERS["typical"], "age": 5}
    bad_job = {**SAMPLE_CUSTOMERS["typical"], "job": "astronaut"}
    bad_history = {**SAMPLE_CUSTOMERS["typical"], "poutcome": "success"}
    bad_pdays = {**SAMPLE_CUSTOMERS["high_chance"], "pdays": 999}
    for bad in (bad_age, bad_job, bad_history, bad_pdays):
        try:
            Customer(**bad)
        except ValidationError:
            continue
        raise AssertionError("Schema phai tu choi du lieu sai")
    assert "job" in category_options()


def check_api():
    from fastapi.testclient import TestClient

    from src.api import app

    client = TestClient(app)
    assert client.get("/health").json() == {"status": "ok"}
    assert "job" in client.get("/options").json()
    good = client.post("/predict", json=SAMPLE_CUSTOMERS["typical"])
    assert good.status_code == 200
    assert 0 <= good.json()["probability"] <= 1
    bad = client.post("/predict", json={**SAMPLE_CUSTOMERS["typical"], "age": 5})
    assert bad.status_code == 422
    print("API: /health, /options, /predict deu dung")


def main():
    check_model_on_samples()
    check_schema_rejects_bad_input()
    try:
        check_api()
    except ImportError:
        print("Chua cai fastapi/httpx nen bo qua phan kiem tra API")
    print("OK")


if __name__ == "__main__":
    main()
