import pandas as pd
import os
import mlflow.pyfunc

PROD_PATH = "models/production_model"
_pipeline = None


def load_artifacts():
    global _pipeline
    try:
        if os.path.exists(PROD_PATH):
            _pipeline = mlflow.pyfunc.load_model(PROD_PATH)
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

    # Access underlying sklearn model for probability
    # mlflow.pyfunc.load_model returns a PyFuncModel.
    # For sklearn flavor, _model_impl is the actual sklearn object.
    try:
        model = _pipeline._model_impl
        prob = model.predict_proba(df)[0][1]
        lbl = int(model.predict(df)[0])
    except AttributeError:
        # Fallback if specific implementation is hidden
        res = _pipeline.predict(df)
        lbl = int(res[0])
        prob = float(lbl)

    return {"risk_probability": round(prob, 4), "is_high_risk": bool(lbl)}
