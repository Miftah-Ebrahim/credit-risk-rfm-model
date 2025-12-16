import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
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
    log_loss,
    matthews_corrcoef,
    confusion_matrix,
)


def main():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Credit_Risk_Model")

    data = pd.read_csv("data/processed/data.csv", index_col=0)
    X = data.drop(columns=["Risk_Label", "Cluster"])
    y = data["Risk_Label"]

    X["ChannelId"] = X["ChannelId"].astype(str)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    prep = ColumnTransformer(
        [
            (
                "num",
                StandardScaler(),
                [
                    "Recency",
                    "Frequency",
                    "Monetary_Total",
                    "Monetary_Mean",
                    "Monetary_Std",
                ],
            ),
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["ChannelId"]),
        ]
    )

    models = {
        "LogReg": (LogisticRegression(random_state=42), {"model__C": [0.1, 1, 10]}),
        "GBM": (
            GradientBoostingClassifier(random_state=42),
            {"model__n_estimators": [50, 100]},
        ),
    }

    best_score, best_run_id = 0, None

    for name, (clf, params) in models.items():
        with mlflow.start_run(run_name=name) as run:
            pipe = Pipeline([("prep", prep), ("model", clf)])
            grid = GridSearchCV(pipe, params, cv=3, scoring="roc_auc").fit(
                X_train, y_train
            )

            est = grid.best_estimator_
            probs = est.predict_proba(X_test)[:, 1]
            preds = est.predict(X_test)

            # Metrics
            metrics = {
                "auc": roc_auc_score(y_test, probs),
                "f1": f1_score(y_test, preds),
                "acc": accuracy_score(y_test, preds),
                "prec": precision_score(y_test, preds),
                "rec": recall_score(y_test, preds),
                "log_loss": log_loss(y_test, probs),
                "mcc": matthews_corrcoef(y_test, preds),
            }

            mlflow.log_metrics(metrics)
            mlflow.log_params(grid.best_params_)

            # Log Confusion Matrix as Artifact
            cm = confusion_matrix(y_test, preds)
            mlflow.log_dict({"confusion_matrix": cm.tolist()}, "confusion_matrix.json")

            # Log Model without registration inside run to avoid clutter
            mlflow.sklearn.log_model(est, "model")

            if metrics["auc"] > best_score:
                best_score = metrics["auc"]
                best_run_id = run.info.run_id

    # Register Best Model and Move to Production
    if best_run_id:
        model_uri = f"runs:/{best_run_id}/model"
        reg_name = "Credit_Risk_Model"
        mv = mlflow.register_model(model_uri, reg_name)

        client = MlflowClient()
        client.transition_model_version_stage(
            name=reg_name,
            version=mv.version,
            stage="Production",
            archive_existing_versions=True,
        )
        print(
            f"Registered Best Model (AUC: {best_score:.4f}) to {reg_name} version {mv.version} -> Production"
        )


if __name__ == "__main__":
    main()
