import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
)
from sklearn.preprocessing import StandardScaler
import joblib
import os
import logging
from src.data_processing import preprocess_pipeline

# Configure Logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)


def train_model(df: pd.DataFrame, target_col: str = "Risk_Label"):
    """
    Trains models and logs validation metrics to MLflow.
    """
    try:
        X = df.drop(columns=[target_col, "Cluster"])  # Drop auxiliary columns
        y = df[target_col]

        # Split Data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )

        # Scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Save Scaler for Inference
        joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))

        # Experiment Setup
        mlflow.set_experiment("Credit_Risk_RFM_Model")

        models_to_train = {
            "Logistic_Regression": {
                "model": LogisticRegression(random_state=42),
                "params": {"C": [0.01, 0.1, 1, 10]},
            },
            "Gradient_Boosting": {
                "model": GradientBoostingClassifier(random_state=42),
                "params": {
                    "n_estimators": [50, 100],
                    "learning_rate": [0.01, 0.1],
                    "max_depth": [3, 5],
                },
            },
        }

        best_overall_auc = 0
        best_overall_model = None
        best_model_name = ""

        for name, config in models_to_train.items():
            with mlflow.start_run(run_name=name):
                logger.info(f"Training {name}...")

                # Grid Search
                grid = GridSearchCV(
                    config["model"],
                    config["params"],
                    cv=3,
                    scoring="roc_auc",
                    n_jobs=-1,
                )
                grid.fit(X_train_scaled, y_train)

                best_model = grid.best_estimator_
                preds = best_model.predict(X_test_scaled)
                probs = best_model.predict_proba(X_test_scaled)[:, 1]

                # Metrics
                acc = accuracy_score(y_test, preds)
                prec = precision_score(y_test, preds)
                rec = recall_score(y_test, preds)
                f1 = f1_score(y_test, preds)
                auc = roc_auc_score(y_test, probs)

                # Log to MLflow
                mlflow.log_params(grid.best_params_)
                mlflow.log_metrics(
                    {
                        "accuracy": acc,
                        "precision": prec,
                        "recall": rec,
                        "f1": f1,
                        "auc": auc,
                    }
                )

                mlflow.sklearn.log_model(best_model, "model")

                logger.info(f"{name} Results - AUC: {auc:.4f}, F1: {f1:.4f}")

                if auc > best_overall_auc:
                    best_overall_auc = auc
                    best_overall_model = best_model
                    best_model_name = name

        # Save Best Model
        if best_overall_model:
            model_path = os.path.join(MODELS_DIR, "best_model.pkl")
            joblib.dump(best_overall_model, model_path)
            logger.info(
                f"Best model ({best_model_name}) saved to {model_path} with AUC: {best_overall_auc:.4f}"
            )

    except Exception as e:
        logger.error(f"Error during training: {e}")
        raise


if __name__ == "__main__":
    # Orchestration
    raw_dir = "data/raw"
    files = [f for f in os.listdir(raw_dir) if f.endswith(".csv")]
    if files:
        raw_path = os.path.join(raw_dir, files[0])
        logger.info("Starting Pipeline...")
        processed_df = preprocess_pipeline(raw_path)
        logger.info("Starting Training...")
        train_model(processed_df)
    else:
        logger.error("No data found.")
