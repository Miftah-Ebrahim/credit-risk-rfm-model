import pytest
import pandas as pd
import numpy as np
import sys
import os

# Ensure src is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_processing import engineer_features, create_proxy_target


@pytest.fixture
def sample_data():
    """Creates a mock DataFrame for testing."""
    data = {
        "TransactionId": ["T1", "T2", "T3", "T4"],
        "CustomerId": ["C1", "C1", "C2", "C2"],
        "Amount": [100.0, 200.0, 500.0, 50.0],
        "TransactionStartTime": pd.to_datetime(
            [
                "2023-01-01 10:00:00",
                "2023-01-02 10:00:00",  # C1 Recency should be relatively low
                "2023-01-01 10:00:00",
                "2023-01-05 10:00:00",  # C2 Recency is lowest (most recent)
            ]
        ),
    }
    return pd.DataFrame(data)


def test_engineer_features(sample_data):
    """Test if aggregation works correctly."""
    res = engineer_features(sample_data)

    assert res.shape == (2, 5)  # 2 Customers, 5 Features
    assert "Recency" in res.columns
    assert "Frequency" in res.columns
    assert "Monetary_Total" in res.columns

    # Check C1 values
    # C1 Total = 300
    assert res.loc["C1", "Monetary_Total"] == 300.0
    # C1 Freq = 2
    assert res.loc["C1", "Frequency"] == 2


def test_proxy_target_creation(sample_data):
    """Test if Risk Label is created."""
    # Pre-process
    processed = engineer_features(sample_data)

    # Run Target Creation
    res = create_proxy_target(processed, n_clusters=2, random_state=42)

    assert "Cluster" in res.columns
    assert "Risk_Label" in res.columns
    assert res["Risk_Label"].isin([0, 1]).all()
