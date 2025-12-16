import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans


def calculate_rfm(df: pd.DataFrame) -> pd.DataFrame:
    if "CustomerId" not in df.columns:
        raise ValueError("CustomerId missing")

    max_date = df["TransactionStartTime"].max()
    agg_rules = {
        "TransactionStartTime": lambda x: (max_date - x.max()).days,
        "TransactionId": "count",
        "Amount": ["sum", "mean", "std"],
        "ChannelId": lambda x: x.mode()[0] if not x.mode().empty else x.iloc[0],
    }

    res = df.groupby("CustomerId").agg(agg_rules)
    res.columns = [
        "Recency",
        "Frequency",
        "Monetary_Total",
        "Monetary_Mean",
        "Monetary_Std",
        "ChannelId",
    ]
    res["Monetary_Std"] = res["Monetary_Std"].fillna(0)
    return res


def assign_risk_label(df: pd.DataFrame, n_clusters: int = 3) -> pd.DataFrame:
    features = ["Recency", "Frequency", "Monetary_Total"]
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[features])

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df["Cluster"] = kmeans.fit_predict(scaled)

    risk_cluster = df.groupby("Cluster")["Recency"].mean().idxmax()
    df["Risk_Label"] = (df["Cluster"] == risk_cluster).astype(int)
    return df


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    if "TransactionStartTime" in df.columns:
        df["TransactionHour"] = df["TransactionStartTime"].dt.hour
        df["TransactionDay"] = df["TransactionStartTime"].dt.day
        df["TransactionMonth"] = df["TransactionStartTime"].dt.month
        df["TransactionYear"] = df["TransactionStartTime"].dt.year
    return df


def calculate_woe_iv(df: pd.DataFrame, feature: str, target: str) -> dict:
    lst = []
    for val in df[feature].unique():
        good = df[(df[feature] == val) & (df[target] == 0)].shape[0]
        bad = df[(df[feature] == val) & (df[target] == 1)].shape[0]
        lst.append({"Value": val, "Good": good, "Bad": bad})

    dset = pd.DataFrame(lst)
    dset["Distr_Good"] = dset["Good"] / dset["Good"].sum()
    dset["Distr_Bad"] = dset["Bad"] / dset["Bad"].sum()
    dset["WoE"] = np.log(dset["Distr_Good"] / dset["Distr_Bad"]).replace(
        {np.inf: 0, -np.inf: 0}
    )
    dset["IV"] = (dset["Distr_Good"] - dset["Distr_Bad"]) * dset["WoE"]

    return {"IV": dset["IV"].sum(), "WoE_Table": dset}
