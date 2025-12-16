import pandas as pd
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

MODEL_DIR = "models"
DATA_PATH = "data/processed/data.csv"
PROD_PATH = "models/production_model"


def main():
    if not os.path.exists(DATA_PATH):
        print(f"No data at {DATA_PATH}")
        return

    df = pd.read_csv(DATA_PATH, index_col=0)
    X = df.drop(columns=["Risk_Label", "Cluster"])
    y = df["Risk_Label"]

    # Feature Config
    num_feats = [
        "Recency",
        "Frequency",
        "Monetary_Total",
        "Monetary_Mean",
        "Monetary_Std",
    ]
    cat_feats = ["ChannelId"]
    X["ChannelId"] = X["ChannelId"].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    preprocessor = ColumnTransformer(
        [
            ("num", StandardScaler(), num_feats),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_feats),
        ]
    )

    mlflow.set_experiment("Credit_Risk_Pipeline")
    best_auc = 0
    best_pipe = None

    models = {
        "LogReg": (LogisticRegression(random_state=42), {"model__C": [0.1, 1, 10]}),
        "GBM": (
            GradientBoostingClassifier(random_state=42),
            {"model__n_estimators": [50, 100], "model__max_depth": [3]},
        ),
    }

    for name, (clf, params) in models.items():
        with mlflow.start_run(run_name=name):
            pipe = Pipeline([("preprocessor", preprocessor), ("model", clf)])
            grid = GridSearchCV(pipe, params, cv=3, scoring="roc_auc")
            grid.fit(X_train, y_train)

            est = grid.best_estimator_
            preds = est.predict(X_test)
            probs = est.predict_proba(X_test)[:, 1]

            metrics = {
                "auc": roc_auc_score(y_test, probs),
                "f1": f1_score(y_test, preds),
                "acc": accuracy_score(y_test, preds),
                "prec": precision_score(y_test, preds),
                "rec": recall_score(y_test, preds),
            }

            print(f"{name}: {metrics}")
            mlflow.log_metrics(metrics)
            mlflow.log_params(grid.best_params_)

            if metrics["auc"] >= best_auc:
                best_auc = metrics["auc"]
                best_pipe = est

    if os.path.exists(PROD_PATH):
        shutil.rmtree(PROD_PATH)
    mlflow.sklearn.save_model(best_pipe, PROD_PATH)
    print(f"Saved Best Model (AUC: {best_auc:.4f}) to {PROD_PATH}")


if __name__ == "__main__":
    main()
