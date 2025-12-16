import pandas as pd
import numpy as np
import os
import logging
from src.features import (
    calculate_rfm,
    assign_risk_label,
    add_temporal_features,
    calculate_woe_iv,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_data(filepath: str) -> pd.DataFrame:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"{filepath} not found")
    return pd.read_csv(filepath)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "TransactionStartTime" in df.columns:
        df["TransactionStartTime"] = pd.to_datetime(df["TransactionStartTime"])

    df.drop_duplicates(inplace=True)
    df = add_temporal_features(df)

    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isnull().any():
            df[col].fillna(df[col].median(), inplace=True)
    return df


def run_pipeline(raw_path: str, output_path: str = "data/processed/data.csv"):
    logger.info("Starting Pipeline...")
    df = clean_data(load_data(raw_path))
    rfm = calculate_rfm(df)
    final = assign_risk_label(rfm)

    for feat in ["Recency", "Frequency", "Monetary_Total"]:
        try:
            temp = final.copy()
            temp[feat + "_Bin"] = pd.qcut(temp[feat], q=4, duplicates="drop")
            iv = calculate_woe_iv(temp, feat + "_Bin", "Risk_Label")["IV"]
            logger.info(f"{feat} IV: {iv:.4f}")
        except:
            pass

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    final.to_csv(output_path)
    logger.info(f"Saved to {output_path}")
    return final


if __name__ == "__main__":
    raw_dir = "data/raw"
    files = [f for f in os.listdir(raw_dir) if f.endswith(".csv")]
    if files:
        run_pipeline(os.path.join(raw_dir, files[0]))
