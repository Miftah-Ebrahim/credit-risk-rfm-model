import joblib
import pandas as pd
import os
import logging

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODELS_DIR = "models"
MODEL_PATH = os.path.join(MODELS_DIR, "best_model.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")


class CreditRiskModel:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.load_artifacts()

    def load_artifacts(self):
        """Loads model and scaler from disk."""
        try:
            if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
                self.model = joblib.load(MODEL_PATH)
                self.scaler = joblib.load(SCALER_PATH)
                logger.info("Model and Scaler loaded successfully.")
            else:
                logger.warning(
                    f"Artifacts not found at {MODELS_DIR}. Predictions will fail until training is run."
                )
        except Exception as e:
            logger.error(f"Error loading artifacts: {e}")

    def predict(self, input_data: dict):
        """
        Predicts credit risk for a single Customer input.
        Input format: {'Recency': int, 'Frequency': int, 'Monetary_Total': float, 'Monetary_Mean': float, 'Monetary_Std': float}
        """
        if not self.model or not self.scaler:
            raise RuntimeError("Model not loaded. Train the model first.")

        try:
            # Convert dict to DataFrame
            df = pd.DataFrame([input_data])

            # Ensure correct order of features (must match training)
            expected_cols = [
                "Recency",
                "Frequency",
                "Monetary_Total",
                "Monetary_Mean",
                "Monetary_Std",
            ]
            df = df[expected_cols]

            # Scale
            scaled_data = self.scaler.transform(df)

            # Predict
            prob = self.model.predict_proba(scaled_data)[0][1]
            label = int(self.model.predict(scaled_data)[0])

            return {"risk_probability": round(prob, 4), "is_high_risk": bool(label)}
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise
