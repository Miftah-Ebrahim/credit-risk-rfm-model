import pandas as pd
import os
import mlflow.sklearn

PRODUCTION_MODEL_PATH = "models/production_model"

_pipeline = None


def load_artifacts():
    """Loads the MLflow pipeline from the production path."""
    global _pipeline
    if os.path.exists(PRODUCTION_MODEL_PATH):
        try:
            _pipeline = mlflow.sklearn.load_model(PRODUCTION_MODEL_PATH)
            return True
        except Exception as e:
            print(f"Failed to load MLflow model: {e}")
            return False
    return False


def predict_risk(data: dict):
    """
    Predicts risk for a single record.
    Data format: {Recency, Frequency, Monetary_Total, ..., ChannelId}
    """
    if _pipeline is None:
        if not load_artifacts():
            raise RuntimeError("Model artifacts not found.")

    # Prepare DataFrame
    df = pd.DataFrame([data])

    # Ensure ChannelId is string for OneHotEncoding
    if "ChannelId" in df.columns:
        df["ChannelId"] = df["ChannelId"].astype(str)

    # Predict (Pipeline handles scaling/encoding)
    prob = _pipeline.predict_proba(df)[0][1]
    is_risk = int(_pipeline.predict(df)[0])

    return {"risk_probability": round(prob, 4), "is_high_risk": bool(is_risk)}
