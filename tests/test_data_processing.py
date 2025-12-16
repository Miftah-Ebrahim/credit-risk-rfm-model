import pytest
import pandas as pd
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.features import calculate_rfm, assign_risk_label, add_temporal_features


@pytest.fixture
def data():
    return pd.DataFrame(
        {
            "CustomerId": ["C1", "C2"],
            "TransactionStartTime": pd.to_datetime(["2023-01-01", "2023-02-01"]),
            "Amount": [100, 200],
            "TransactionId": ["T1", "T2"],
            "ChannelId": ["Web", "App"],
        }
    )


def test_rfm(data):
    res = calculate_rfm(data)
    assert res.shape == (2, 6)
    assert "ChannelId" in res.columns


def test_risk_label(data):
    rfm = calculate_rfm(data)
    res = assign_risk_label(rfm, n_clusters=2)  # k=2 for small sample
    assert "Risk_Label" in res.columns


def test_temporal(data):
    res = add_temporal_features(data)
    assert "TransactionYear" in res.columns
