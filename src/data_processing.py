import pandas as pd
import numpy as np
import os
import logging
from src.features import calculate_rfm, assign_risk_label

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_data(filepath: str) -> pd.DataFrame:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    return pd.read_csv(filepath)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Types
    if "TransactionStartTime" in df.columns:
        df["TransactionStartTime"] = pd.to_datetime(df["TransactionStartTime"])

    # Dedup
    df.drop_duplicates(inplace=True)

    # Impute Numerics
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isnull().any():
            df[col].fillna(df[col].median(), inplace=True)

    return df


def run_pipeline(raw_path: str, output_path: str = "data/processed/data.csv"):
    """Orchestrates the data pipeline."""
    logger.info("Loading Data...")
    df = load_data(raw_path)

    logger.info("Cleaning Data...")
    df = clean_data(df)

    logger.info("Engineering Features...")
    rfm_df = calculate_rfm(df)

    logger.info("Creating Proxy Target...")
    final_df = assign_risk_label(rfm_df)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    final_df.to_csv(output_path)
    logger.info(f"Saved to {output_path}")

    return final_df


if __name__ == "__main__":
    # Auto-detection for standalone run
    raw_dir = "data/raw"
    files = [f for f in os.listdir(raw_dir) if f.endswith(".csv")]
    if files:
        run_pipeline(os.path.join(raw_dir, files[0]))
