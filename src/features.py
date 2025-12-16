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
        "ChannelId": lambda x: x.mode()[0]
        if not x.mode().empty
        else x.iloc[0],  # Categorical Mode
    }

    customer_df = df.groupby("CustomerId").agg(agg_rules)
    customer_df.columns = [
        "Recency",
        "Frequency",
        "Monetary_Total",
        "Monetary_Mean",
        "Monetary_Std",
        "ChannelId",
    ]
    customer_df["Monetary_Std"] = customer_df["Monetary_Std"].fillna(0)

    return customer_df


def assign_risk_label(df: pd.DataFrame, n_clusters: int = 3) -> pd.DataFrame:
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
    # With 3 clusters, we still isolate the "worst" one as High Risk (1), others typically (0)
    risk_cluster = df.groupby("Cluster")["Recency"].mean().idxmax()
    df["Risk_Label"] = (df["Cluster"] == risk_cluster).astype(int)

    return df


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts temporal features from TransactionStartTime.
    """
    if "TransactionStartTime" in df.columns:
        df["TransactionHour"] = df["TransactionStartTime"].dt.hour
        df["TransactionDay"] = df["TransactionStartTime"].dt.day
        df["TransactionMonth"] = df["TransactionStartTime"].dt.month
        df["TransactionYear"] = df["TransactionStartTime"].dt.year
    return df


def calculate_woe_iv(df: pd.DataFrame, feature: str, target: str) -> dict:
    """
    Calculates Weight of Evidence (WoE) and Information Value (IV) for a feature.
    """
    lst = []
    for i in range(df[feature].nunique()):
        val = list(df[feature].unique())[i]
        lst.append(
            {
                "Value": val,
                "All": df[df[feature] == val].count()[feature],
                "Good": df[(df[feature] == val) & (df[target] == 0)].count()[feature],
                "Bad": df[(df[feature] == val) & (df[target] == 1)].count()[feature],
            }
        )

    dset = pd.DataFrame(lst)
    dset["Distr_Good"] = dset["Good"] / dset["Good"].sum()
    dset["Distr_Bad"] = dset["Bad"] / dset["Bad"].sum()
    dset["WoE"] = np.log(dset["Distr_Good"] / dset["Distr_Bad"])
    dset = dset.replace({"WoE": {np.inf: 0, -np.inf: 0}})
    dset["IV"] = (dset["Distr_Good"] - dset["Distr_Bad"]) * dset["WoE"]

    iv = dset["IV"].sum()
    return {"IV": iv, "WoE_Table": dset}
