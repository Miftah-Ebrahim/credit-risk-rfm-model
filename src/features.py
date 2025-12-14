import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans


def calculate_rfm(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates transaction data to Customer Level (Recency, Frequency, Monetary).
    """
    if "CustomerId" not in df.columns:
        raise ValueError("CustomerId missing")

    max_date = df["TransactionStartTime"].max()

    # Aggregations
    agg_rules = {
        "TransactionStartTime": lambda x: (max_date - x.max()).days,
        "TransactionId": "count",
        "Amount": ["sum", "mean", "std"],
    }

    customer_df = df.groupby("CustomerId").agg(agg_rules)
    customer_df.columns = [
        "Recency",
        "Frequency",
        "Monetary_Total",
        "Monetary_Mean",
        "Monetary_Std",
    ]
    customer_df["Monetary_Std"] = customer_df["Monetary_Std"].fillna(0)

    return customer_df


def assign_risk_label(df: pd.DataFrame, n_clusters: int = 2) -> pd.DataFrame:
    """
    Assigns 'Risk_Label' using KMeans clustering on RFM features.
    High Risk = High Recency cluster.
    """
    features = ["Recency", "Frequency", "Monetary_Total"]
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[features])

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df["Cluster"] = kmeans.fit_predict(scaled_data)

    # Heuristic: Cluster with highest avg Recency is 'High Risk' (Churned/Dormant)
    risk_cluster = df.groupby("Cluster")["Recency"].mean().idxmax()
    df["Risk_Label"] = (df["Cluster"] == risk_cluster).astype(int)

    return df
