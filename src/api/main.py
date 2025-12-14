from fastapi import FastAPI, HTTPException
from src.api.pydantic_models import CustomerData, PredictionResponse
from src.predict import CreditRiskModel
import uvicorn
import logging

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize App
app = FastAPI(title="Credit Risk RFM API", version="1.0.0")

# Initialize Model (Lazy loading or startup)
model_service = CreditRiskModel()


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "model_loaded": model_service.model is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict_risk(data: CustomerData):
    """
    Predicts credit risk based on RFM features.
    """
    try:
        result = model_service.predict(data.dict())
        result["status"] = "success"
        return result
    except RuntimeError as e:
        # Model not ready
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"API Error: {e}")
        raise HTTPException(
            status_code=400, detail="Prediction failed. Verify input data."
        )


if __name__ == "__main__":
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
