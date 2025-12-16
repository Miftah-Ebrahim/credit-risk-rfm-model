from fastapi import FastAPI, HTTPException
from api.schemas import CustomerData, PredictionResponse
from src.predict import predict_risk, load_artifacts

app = FastAPI(title="Credit Risk API")


@app.on_event("startup")
def startup():
    load_artifacts()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(data: CustomerData):
    try:
        return predict_risk(data.dict())
    except Exception as e:
        raise HTTPException(500, str(e))
