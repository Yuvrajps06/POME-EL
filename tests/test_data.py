"""Tests for Phase 1 data pipeline."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.data.feature_engineering import (
    add_lag_features,
    add_rolling_features,
    chronological_split,
)
from src.data.loader import load_calendar
from src.utils.config import DATA_PROCESSED_DIR, DATA_SAMPLE_DIR


@pytest.fixture(scope="module")
def sample_data() -> pd.DataFrame:
    """Load sample processed dataset for test execution."""
    path = DATA_SAMPLE_DIR / "m5_sample.csv"
    if not path.exists():
        path = DATA_PROCESSED_DIR / "m5_processed.parquet"
    if str(path).endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def test_calendar_date_mapping() -> None:
    """Verify that d_1 maps to 2011-01-29 and sequence is continuous."""
    cal = load_calendar()
    row_1 = cal[cal["d"] == "d_1"].iloc[0]
    assert pd.to_datetime(row_1["date"]) == pd.Timestamp("2011-01-29")
    assert row_1["wm_yr_wk"] == 11101


def test_no_negative_prices(sample_data: pd.DataFrame) -> None:
    """Verify sell_price has no negative or zero values."""
    assert (sample_data["sell_price"] > 0).all()
    assert (sample_data["log_price"].notna()).all()


def test_no_future_leakage_in_lags() -> None:
    """Verify lag sales strictly equal shifted past sales with no lookahead."""
    dates = pd.date_range("2020-01-01", periods=10, freq="D")
    df = pd.DataFrame({
        "item_id": ["ITEM_A"] * 10,
        "store_id": ["STORE_1"] * 10,
        "date": dates,
        "sales": [float(i) for i in range(10)],
    })

    lagged = add_lag_features(df, lags=[1, 7])
    # lag 1 at index 0 should be NaN
    assert np.isnan(lagged.loc[0, "lag_sales_1"])
    # lag 1 at index 1 should be sales at index 0 (0.0)
    assert lagged.loc[1, "lag_sales_1"] == 0.0
    # lag 1 at index 5 should be sales at index 4 (4.0)
    assert lagged.loc[5, "lag_sales_1"] == 4.0
    # lag 7 at index 7 should be sales at index 0 (0.0)
    assert lagged.loc[7, "lag_sales_7"] == 0.0


def test_no_future_leakage_in_rolling() -> None:
    """Verify rolling calculations shift before rolling to avoid target leakage."""
    dates = pd.date_range("2020-01-01", periods=10, freq="D")
    df = pd.DataFrame({
        "item_id": ["ITEM_A"] * 10,
        "store_id": ["STORE_1"] * 10,
        "date": dates,
        "sales": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0],
    })

    rolled = add_rolling_features(df, windows=[7])
    # At index 1, shifted series only has sales at index 0 (10.0)
    # Target sales at index 1 is 20.0, which should NOT affect rolling_mean at index 1
    assert rolled.loc[1, "rolling_mean_7"] == 10.0
    # At index 2, rolling mean of shifted should be (10 + 20) / 2 = 15.0
    assert rolled.loc[2, "rolling_mean_7"] == 15.0


def test_chronological_split_ordering() -> None:
    """Verify chronological split guarantees train < val < test dates."""
    dates = pd.date_range("2020-01-01", periods=100, freq="D")
    df = pd.DataFrame({
        "date": dates,
        "sales": np.ones(100),
    })

    train, val, test = chronological_split(df, test_days=28, val_days=28)

    assert len(test["date"].unique()) == 28
    assert len(val["date"].unique()) == 28
    assert len(train["date"].unique()) == 100 - 56

    train_max_date = train["date"].max()
    val_min_date = val["date"].min()
    val_max_date = val["date"].max()
    test_min_date = test["date"].min()

    assert train_max_date < val_min_date
    assert val_max_date < test_min_date
