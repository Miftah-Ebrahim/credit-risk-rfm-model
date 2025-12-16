import pandas as pd
import mlflow.pyfunc
import mlflow

# Point to SQLite registry if running locally
mlflow.set_tracking_uri("sqlite:///mlflow.db")

MODEL_URI = "models:/Credit_Risk_Model/Production"
_model = None


def load_model():
    global _model
    try:
        _model = mlflow.pyfunc.load_model(MODEL_URI)
    except:
        pass


def predict_risk(data: dict):
    if not _model:
        load_model()
    if not _model:
        raise RuntimeError("Model unavailable or registry Empty")

    df = pd.DataFrame([data])
    if "ChannelId" in df:
        df["ChannelId"] = df["ChannelId"].astype(str)

    # Generic PyFunc handling with fallback for sklearn probing
    try:
        est = _model._model_impl
        prob = est.predict_proba(df)[0][1]
        pred = int(est.predict(df)[0])
    except:
        res = _model.predict(df)
        pred = int(res[0])
        prob = float(pred)  # Fallback if proba not available

    return {"risk_probability": round(prob, 4), "is_high_risk": bool(pred)}
