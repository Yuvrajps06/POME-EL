"""Feature engineering module for retail demand modeling.

Computes price dynamics, leakage-free lagged sales, rolling aggregates,
calendar and SNAP indicators, log transformations, and chronological data splits.
"""

from typing import List, Tuple
import numpy as np
import pandas as pd


def add_lag_features(
    df: pd.DataFrame,
    lags: List[int] = [1, 7, 14, 28],
) -> pd.DataFrame:
    """Add lagged sales features strictly within each (item_id, store_id) group.

    Args:
        df: DataFrame sorted by (item_id, store_id, date).
        lags: List of day lags.

    Returns:
        DataFrame with lag_sales_{k} columns.
    """
    res = df.copy()
    grouped = res.groupby(["item_id", "store_id"])["sales"]

    for lag in lags:
        res[f"lag_sales_{lag}"] = grouped.shift(lag)

    return res


def add_rolling_features(
    df: pd.DataFrame,
    windows: List[int] = [7, 28],
) -> pd.DataFrame:
    """Add rolling mean and standard deviation features avoiding future leakage.

    Crucially, shifts by 1 day before computing rolling statistics so current
    day sales (target) is never visible in feature inputs.

    Args:
        df: DataFrame sorted by (item_id, store_id, date).
        windows: List of window day lengths.

    Returns:
        DataFrame with rolling_mean_{w} and rolling_std_{w} columns.
    """
    res = df.copy()
    grouped = res.groupby(["item_id", "store_id"])["sales"]

    for w in windows:
        # Shift 1 first to avoid target leakage, then roll
        shifted = grouped.shift(1)
        res[f"rolling_mean_{w}"] = shifted.groupby(
            [res["item_id"], res["store_id"]]
        ).transform(lambda s: s.rolling(w, min_periods=1).mean())

        res[f"rolling_std_{w}"] = shifted.groupby(
            [res["item_id"], res["store_id"]]
        ).transform(lambda s: s.rolling(w, min_periods=2).std().fillna(0.0))

    return res


def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer price dynamics and relative price indicators.

    Computes percentage price change and ratio relative to item's mean price.

    Args:
        df: DataFrame containing sell_price, item_id, store_id.

    Returns:
        DataFrame with price_change_pct and price_rel_mean columns.
    """
    res = df.copy()

    # Percentage change in price from previous day
    res["price_change_pct"] = (
        res.groupby(["item_id", "store_id"])["sell_price"]
        .pct_change()
        .fillna(0.0)
    )

    # Item mean price across history
    item_mean = res.groupby("item_id")["sell_price"].transform("mean")
    res["price_rel_mean"] = res["sell_price"] / item_mean

    # Discount / promotion flag: price strictly below item average
    res["is_discounted"] = (res["sell_price"] < item_mean).astype(int)

    return res


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract day-of-week, month, and binary event flags.

    Args:
        df: DataFrame containing date and calendar metadata.

    Returns:
        DataFrame enriched with temporal indicator columns.
    """
    res = df.copy()
    dates = pd.to_datetime(res["date"])

    res["day_of_week"] = dates.dt.dayofweek
    res["day_of_month"] = dates.dt.day
    res["month"] = dates.dt.month
    res["year"] = dates.dt.year
    res["is_weekend"] = res["day_of_week"].isin([5, 6]).astype(int)

    # Event / holiday active indicator
    if "event_name_1" in res.columns:
        res["has_event"] = res["event_name_1"].notna().astype(int)
    else:
        res["has_event"] = 0

    return res


def add_log_transforms(df: pd.DataFrame) -> pd.DataFrame:
    """Compute natural log of price and log1p of sales for log-log demand models.

    Args:
        df: DataFrame with sell_price and sales.

    Returns:
        DataFrame with log_price and log_sales columns.
    """
    res = df.copy()
    # Guard against zero/negative prices
    clipped_price = np.maximum(res["sell_price"].values, 1e-4)
    res["log_price"] = np.log(clipped_price).astype("float32")
    res["log_sales"] = np.log1p(np.maximum(res["sales"].values, 0.0)).astype("float32")
    return res


def build_feature_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """Execute end-to-end feature engineering pipeline.

    Args:
        df: Clean preprocessed DataFrame.

    Returns:
        Feature-engineered DataFrame sorted chronologically.
    """
    out = df.sort_values(by=["item_id", "store_id", "date"]).reset_index(drop=True)
    out = add_price_features(out)
    out = add_lag_features(out, lags=[1, 7, 14, 28])
    out = add_rolling_features(out, windows=[7, 28])
    out = add_calendar_features(out)
    out = add_log_transforms(out)

    return out


def chronological_split(
    df: pd.DataFrame,
    test_days: int = 28,
    val_days: int = 28,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split time-series data chronologically into train, validation, and test sets.

    Test set comprises the last `test_days`.
    Validation set comprises the `val_days` preceding the test set.
    Training set comprises all preceding history.

    Args:
        df: Feature-engineered DataFrame with a 'date' column.
        test_days: Number of days allocated to test evaluation (default: 28).
        val_days: Number of days allocated to validation (default: 28).

    Returns:
        Tuple of (train_df, val_df, test_df).
    """
    all_dates = sorted(df["date"].unique())
    n_dates = len(all_dates)

    if n_dates <= (test_days + val_days):
        raise ValueError(
            f"Insufficient date range ({n_dates} days) for split: "
            f"requires > {test_days + val_days} days."
        )

    test_start_date = all_dates[-(test_days)]
    val_start_date = all_dates[-(test_days + val_days)]

    train_df = df[df["date"] < val_start_date].copy().reset_index(drop=True)
    val_df = (
        df[(df["date"] >= val_start_date) & (df["date"] < test_start_date)]
        .copy()
        .reset_index(drop=True)
    )
    test_df = df[df["date"] >= test_start_date].copy().reset_index(drop=True)

    return train_df, val_df, test_df
