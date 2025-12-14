import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
import os
import logging

# Configure Logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_data(filepath: str) -> pd.DataFrame:
    """
    Loads raw data from a CSV file.
    """
    try:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")

        df = pd.read_csv(filepath)
        logger.info(f"Data loaded successfully from {filepath}. Shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs basic data cleaning: typing, missing values (imputation).
    """
    try:
        df = df.copy()

        # Convert Timestamp
        if "TransactionStartTime" in df.columns:
            df["TransactionStartTime"] = pd.to_datetime(df["TransactionStartTime"])

        # Drop duplicates
        initial_count = len(df)
        df.drop_duplicates(inplace=True)
        if len(df) < initial_count:
            logger.info(f"Dropped {initial_count - len(df)} duplicate rows.")

        # Basic Imputation (Median for numerical, Mode for categorical)
        # Assuming minimal missingness based on EDA
        for col in df.select_dtypes(include=[np.number]).columns:
            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(df[col].median())

        logger.info("Data cleaning completed.")
        return df
    except Exception as e:
        logger.error(f"Error in data cleaning: {e}")
        raise


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates transaction data to Customer Level (RFM + Stats).
    """
    try:
        if "CustomerId" not in df.columns:
            raise ValueError("CustomerId column missing.")

        # Recency Reference Date
        max_date = df["TransactionStartTime"].max()

        # Aggregation Dictionary
        agg_rules = {
            "TransactionStartTime": lambda x: (max_date - x.max()).days,  # Recency
            "TransactionId": "count",  # Frequency
            "Amount": ["sum", "mean", "std"],  # Monetary + Stats
        }

        # GroupBy Customer
        customer_df = df.groupby("CustomerId").agg(agg_rules)

        # Flatten MultiIndex Columns
        customer_df.columns = [
            "Recency",
            "Frequency",
            "Monetary_Total",
            "Monetary_Mean",
            "Monetary_Std",
        ]

        # Fill NaN Std (for users with 1 transaction)
        customer_df["Monetary_Std"] = customer_df["Monetary_Std"].fillna(0)

        logger.info(
            f"Feature engineering completed. Customer-level shape: {customer_df.shape}"
        )
        return customer_df
    except Exception as e:
        logger.error(f"Error in feature engineering: {e}")
        raise


def create_proxy_target(
    df: pd.DataFrame, n_clusters: int = 2, random_state: int = 42
) -> pd.DataFrame:
    """
    Uses KMeans to cluster customers based on RFM and create a 'Risk_Label'.
    Assumption: Cluster with lower Value/Freq and Higher Recency is 'High Risk'.
    """
    try:
        # Features for Clustering
        rfm_cols = ["Recency", "Frequency", "Monetary_Total"]

        # Scaling for KMeans
        scaler = MinMaxScaler()
        rfm_scaled = scaler.fit_transform(df[rfm_cols])

        # KMeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        df["Cluster"] = kmeans.fit_predict(rfm_scaled)

        # Identify High Risk Cluster
        # We assume High Risk = High Recency, Low Frequency, Low Monetary
        cluster_stats = df.groupby("Cluster")[rfm_cols].mean()

        # Simple heuristic: The cluster with highest Recency is High Risk
        high_risk_cluster = cluster_stats["Recency"].idxmax()

        df["Risk_Label"] = (df["Cluster"] == high_risk_cluster).astype(int)
        logger.info(f"Proxy target created. High Risk Cluster: {high_risk_cluster}")

        return df
    except Exception as e:
        logger.error(f"Error creating proxy target: {e}")
        raise


def preprocess_pipeline(raw_filepath: str) -> pd.DataFrame:
    """
    End-to-End Processing Pipeline.
    """
    df = load_data(raw_filepath)
    df = clean_data(df)
    customer_df = engineer_features(df)
    final_df = create_proxy_target(customer_df)

    # Save processed data for debugging/checking
    output_path = "data/processed/model_ready_data.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    final_df.to_csv(output_path)
    logger.info(f"Processed data saved to {output_path}")

    return final_df


if __name__ == "__main__":
    # Example Usage check
    # Assumes data/raw/data.csv exists per previous context
    import os

    raw_dir = "data/raw"
    files = [f for f in os.listdir(raw_dir) if f.endswith(".csv")]
    if files:
        path = os.path.join(raw_dir, files[0])
        preprocess_pipeline(path)
    else:
        logger.warning("No raw data found to test pipeline.")
