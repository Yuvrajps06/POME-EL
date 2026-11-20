"""Data loading module for M5 Walmart retail dataset.

Provides memory-efficient loading with type downcasting, selective column
filtering, and item ranking utility.
"""

from pathlib import Path
from typing import List, Optional, Union
import numpy as np
import pandas as pd

from src.utils.config import (
    DATA_RAW_DIR,
    DEFAULT_CATEGORY,
    DEFAULT_ITEMS,
    DEFAULT_STORES,
)


def load_calendar(
    path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """Load calendar dataset mapping days (d_1..) to dates, weeks, and events.

    Args:
        path: Path to calendar.csv. Defaults to DATA_RAW_DIR / 'calendar.csv'.

    Returns:
        pd.DataFrame with calendar attributes, including parsed dates.
    """
    file_path = Path(path) if path else DATA_RAW_DIR / "calendar.csv"
    if not file_path.exists():
        raise FileNotFoundError(f"Calendar file not found at {file_path}")

    dtypes = {
        "wm_yr_wk": "int32",
        "weekday": "category",
        "wday": "int8",
        "month": "int8",
        "year": "int16",
        "event_name_1": "category",
        "event_type_1": "category",
        "event_name_2": "category",
        "event_type_2": "category",
        "snap_CA": "int8",
        "snap_TX": "int8",
        "snap_WI": "int8",
    }

    df = pd.read_csv(file_path, dtype=dtypes)
    df["date"] = pd.to_datetime(df["date"])

    # Ensure 'd' column exists mapping to row index (d_1..d_1969)
    if "d" not in df.columns:
        df["d"] = [f"d_{i + 1}" for i in range(len(df))]
    df["d"] = df["d"].astype("category")

    return df


def load_prices(
    path: Optional[Union[str, Path]] = None,
    store_ids: Optional[List[str]] = None,
    item_ids: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Load weekly sell prices with optional filtering.

    Args:
        path: Path to sell_prices.csv. Defaults to DATA_RAW_DIR / 'sell_prices.csv'.
        store_ids: List of store IDs to filter (e.g. ['CA_1', 'CA_2']).
        item_ids: List of item IDs to filter.

    Returns:
        pd.DataFrame containing weekly item prices per store.
    """
    file_path = Path(path) if path else DATA_RAW_DIR / "sell_prices.csv"
    if not file_path.exists():
        raise FileNotFoundError(f"Prices file not found at {file_path}")

    dtypes = {
        "store_id": "category",
        "item_id": "category",
        "wm_yr_wk": "int32",
        "sell_price": "float32",
    }

    df = pd.read_csv(file_path, dtype=dtypes)

    if store_ids:
        df = df[df["store_id"].isin(store_ids)]
    if item_ids:
        df = df[df["item_id"].isin(item_ids)]

    return df.reset_index(drop=True)


def load_sales(
    path: Optional[Union[str, Path]] = None,
    category: Optional[str] = None,
    store_ids: Optional[List[str]] = None,
    item_ids: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Load wide sales dataset with optional row filtering.

    Args:
        path: Path to sales_train_validation.csv.
        category: Category prefix or ID (e.g. 'FOODS').
        store_ids: Filter by list of store IDs.
        item_ids: Filter by list of item IDs.

    Returns:
        pd.DataFrame with wide sales records (d_1 to d_1913).
    """
    file_path = Path(path) if path else DATA_RAW_DIR / "sales_train_validation.csv"
    if not file_path.exists():
        raise FileNotFoundError(f"Sales file not found at {file_path}")

    # Inspect first row to prepare dtypes
    preview = pd.read_csv(file_path, nrows=1)
    d_cols = [c for c in preview.columns if c.startswith("d_")]
    id_cols = [c for c in preview.columns if not c.startswith("d_")]

    dtypes = {c: "category" for c in id_cols}
    dtypes.update({c: "int16" for c in d_cols})

    df = pd.read_csv(file_path, dtype=dtypes)

    if category:
        if "cat_id" in df.columns:
            df = df[df["cat_id"] == category]
        else:
            df = df[df["item_id"].str.startswith(category)]

    if store_ids:
        df = df[df["store_id"].isin(store_ids)]

    if item_ids:
        df = df[df["item_id"].isin(item_ids)]

    return df.reset_index(drop=True)


def rank_candidate_items(
    prices_df: pd.DataFrame,
    sales_df: pd.DataFrame,
    stores: Optional[List[str]] = None,
    top_n: int = 20,
) -> pd.DataFrame:
    """Rank candidate items by distinct price count and low zero-sales share.

    Args:
        prices_df: Filtered sell_prices DataFrame.
        sales_df: Wide sales DataFrame.
        stores: Stores to consider. Defaults to DEFAULT_STORES.
        top_n: Number of top candidate items to return.

    Returns:
        pd.DataFrame containing ranked items with stats.
    """
    target_stores = stores or DEFAULT_STORES

    # Price variation across stores
    p_sub = prices_df[prices_df["store_id"].isin(target_stores)]
    price_stats = p_sub.groupby(["item_id", "store_id"], observed=True)[
        "sell_price"
    ].agg(
        n_distinct="nunique",
        min_price="min",
        max_price="max",
        mean_price="mean",
        std_price="std",
    ).reset_index()

    p_pivot = price_stats.pivot(
        index="item_id", columns="store_id", values="n_distinct"
    ).dropna()

    # Sales metrics
    d_cols = [c for c in sales_df.columns if c.startswith("d_")]
    s_sub = sales_df[
        (sales_df["store_id"].isin(target_stores))
        & (sales_df["item_id"].isin(p_pivot.index))
    ].copy()

    s_sub["zero_share"] = (s_sub[d_cols] == 0).mean(axis=1)
    s_sub["daily_mean"] = s_sub[d_cols].mean(axis=1)
    s_sub["total_sales"] = s_sub[d_cols].sum(axis=1)

    agg_sales = s_sub.groupby("item_id", observed=True).agg(
        mean_zero_share=("zero_share", "mean"),
        avg_daily_sales=("daily_mean", "mean"),
        total_sales=("total_sales", "sum"),
    ).reset_index()

    for st in target_stores:
        if st in p_pivot.columns:
            agg_sales[f"{st}_distinct_prices"] = agg_sales["item_id"].map(p_pivot[st])

    distinct_cols = [f"{st}_distinct_prices" for st in target_stores if f"{st}_distinct_prices" in agg_sales.columns]
    agg_sales["min_distinct_prices"] = agg_sales[distinct_cols].min(axis=1)

    # Composite ranking score: prioritize high distinct prices and low zero-share
    agg_sales["rank_score"] = (
        agg_sales["min_distinct_prices"]
        * (1.0 - agg_sales["mean_zero_share"])
        * np.log1p(agg_sales["avg_daily_sales"])
    )

    ranked = agg_sales.sort_values(by="rank_score", ascending=False).reset_index(drop=True)
    return ranked.head(top_n)
