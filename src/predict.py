import pandas as pd
import os
import mlflow.sklearn

PROD_PATH = "models/production_model"
_pipeline = None


def load_artifacts():
    global _pipeline
    try:
        if os.path.exists(PROD_PATH):
            _pipeline = mlflow.sklearn.load_model(PROD_PATH)
            return True
    except:
        pass
    return False


def predict_risk(data: dict):
    if _pipeline is None and not load_artifacts():
        raise RuntimeError("Model not loaded")

    df = pd.DataFrame([data])
    if "ChannelId" in df.columns:
        df["ChannelId"] = df["ChannelId"].astype(str)

    prob = _pipeline.predict_proba(df)[0][1]
    lbl = int(_pipeline.predict(df)[0])

    return {"risk_probability": round(prob, 4), "is_high_risk": bool(lbl)}
