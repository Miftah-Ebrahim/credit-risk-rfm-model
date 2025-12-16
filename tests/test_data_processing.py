import pytest
import pandas as pd
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.features import calculate_rfm, assign_risk_label, add_temporal_features


@pytest.fixture
def sample_data():
    data = {
        "TransactionId": ["T1", "T2", "T3"],
        "CustomerId": ["C1", "C1", "C2"],
        "Amount": [100.0, 200.0, 500.0],
        "TransactionStartTime": pd.to_datetime(
            ["2023-01-01 10:00:00", "2023-01-02 12:30:00", "2023-01-05 15:45:00"]
        ),
        "ChannelId": ["Web", "Web", "Mobile"],
    }
    return pd.DataFrame(data)


def test_calculate_rfm(sample_data):
    res = calculate_rfm(sample_data)
    # Recency, Freq, M_Total, M_Mean, M_Std, ChannelId = 6 columns
    assert res.shape == (2, 6)
    assert "ChannelId" in res.columns
    # C1 Mode Channel = Web
    assert res.loc["C1", "ChannelId"] == "Web"


def test_assign_risk_label(sample_data):
    rfm = calculate_rfm(sample_data)
    # Using 3 clusters now, but requires >3 samples for meaningful k=3 usually?
    # Logic handles n_samples < n_clusters by raising error or warning in KMeans,
    # but for unit test with 2 samples, we should use n_clusters=1 or 2 to avoid error.
    # Features code defaults to 3, let's override for test data of size 2.
    res = assign_risk_label(rfm, n_clusters=2)
    assert "Risk_Label" in res.columns
    assert res["Risk_Label"].isin([0, 1]).all()


def test_add_temporal_features(sample_data):
    res = add_temporal_features(sample_data)
    assert "TransactionHour" in res.columns
