from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from src.predict import load_pipeline, predict_customer
from src.schema import Customer, Prediction, category_options

WEB_PAGE = Path(__file__).resolve().parents[1] / "web" / "index.html"

app = FastAPI(title="Bank Marketing - du doan khach dang ky tien gui co ky han")


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(WEB_PAGE)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/options")
def options():
    return category_options()


@app.post("/predict", response_model=Prediction)
def predict(customer: Customer):
    try:
        load_pipeline()
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Chua co file models/pipeline.joblib")
    return predict_customer(customer.model_dump())
