import pandas as pd
import numpy as np
import joblib
import os
import shutil
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
)

# Constants
MODEL_DIR = "models"
DATA_PATH = "data/processed/data.csv"
PRODUCTION_MODEL_PATH = "models/production_model"


def main():
    if not os.path.exists(DATA_PATH):
        print(f"Data not found at {DATA_PATH}. Run pipeline first.")
        return

    # Load Data
    df = pd.read_csv(DATA_PATH, index_col=0)  # Index=CustomerId

    # Define Target and Features
    target_col = "Risk_Label"
    drop_cols = ["Risk_Label", "Cluster"]

    X = df.drop(columns=drop_cols)
    y = df[target_col]

    # Feature Definition
    numeric_features = [
        "Recency",
        "Frequency",
        "Monetary_Total",
        "Monetary_Mean",
        "Monetary_Std",
    ]
    categorical_features = ["ChannelId"]

    # Ensure ChannelId is treated as string/category
    X["ChannelId"] = X["ChannelId"].astype(str)

    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Preprocessing Pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    # MLflow Setup
    mlflow.set_experiment("Credit_Risk_Pipeline")
    best_auc = 0
    best_pipe = None

    models = {
        "LogisticReg": (
            LogisticRegression(random_state=42),
            {"model__C": [0.1, 1, 10]},
        ),
        "GradientBoost": (
            GradientBoostingClassifier(random_state=42),
            {"model__n_estimators": [50, 100], "model__max_depth": [3]},
        ),
    }

    for name, (classifier, params) in models.items():
        with mlflow.start_run(run_name=name):
            # Create Full Pipeline
            pipe = Pipeline(
                steps=[("preprocessor", preprocessor), ("model", classifier)]
            )

            # Grid Search (params must be prefixed with model__)
            grid = GridSearchCV(pipe, params, cv=3, scoring="roc_auc")
            grid.fit(X_train, y_train)

            # Eval
            best_estimator = grid.best_estimator_
            preds = best_estimator.predict(X_test)
            probs = best_estimator.predict_proba(X_test)[:, 1]

            auc = roc_auc_score(y_test, probs)
            f1 = f1_score(y_test, preds)
            acc = accuracy_score(y_test, preds)
            prec = precision_score(y_test, preds)
            rec = recall_score(y_test, preds)

            print(f"{name} -> AUC: {auc:.4f}, F1: {f1:.4f}")

            mlflow.log_metrics(
                {"auc": auc, "f1": f1, "acc": acc, "prec": prec, "rec": rec}
            )
            mlflow.log_params(grid.best_params_)

            if auc > best_auc:
                best_auc = auc
                best_pipe = best_estimator

    # Save Best Pipeline via MLflow Native Format for easy loading
    if os.path.exists(PRODUCTION_MODEL_PATH):
        shutil.rmtree(PRODUCTION_MODEL_PATH)

    mlflow.sklearn.save_model(best_pipe, PRODUCTION_MODEL_PATH)
    print(f"Best pipeline saved to {PRODUCTION_MODEL_PATH} with AUC: {best_auc:.4f}")


if __name__ == "__main__":
    main()
