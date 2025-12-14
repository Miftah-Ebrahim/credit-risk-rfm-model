from fastapi import FastAPI, HTTPException
from api.schemas import CustomerData, PredictionResponse
from src.predict import predict_risk, load_artifacts

app = FastAPI(title="RFM Credit Risk API")


@app.on_event("startup")
def startup_event():
    load_artifacts()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(data: CustomerData):
    try:
        result = predict_risk(data.dict())
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
