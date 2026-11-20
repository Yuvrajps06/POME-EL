"""Preprocessing pipeline for M5 retail data.

Converts wide sales tables to long time-series format, merges calendar
and pricing dimensions, eliminates unpriced shelf-absent rows, and flags outliers.
"""

from typing import List, Optional
import numpy as np
import pandas as pd

from src.data.loader import load_calendar, load_prices, load_sales
from src.utils.config import (
    DEFAULT_CATEGORY,
    DEFAULT_ITEMS,
    DEFAULT_STORES,
)


def melt_sales_to_long(sales_df: pd.DataFrame) -> pd.DataFrame:
    """Melt wide daily sales columns (d_1..d_1913) into a tidy long format.

    Args:
        sales_df: Wide sales DataFrame with id, item_id, store_id, and d_* columns.

    Returns:
        pd.DataFrame with columns: [id, item_id, dept_id, cat_id, store_id, state_id, d, sales].
    """
    id_vars = [c for c in sales_df.columns if not c.startswith("d_")]
    value_vars = [c for c in sales_df.columns if c.startswith("d_")]

    long_df = pd.melt(
        sales_df,
        id_vars=id_vars,
        value_vars=value_vars,
        var_name="d",
        value_name="sales",
    )

    long_df["sales"] = long_df["sales"].astype("float32")
    long_df["d"] = long_df["d"].astype(str)
    return long_df


def clean_and_merge(
    sales_long: pd.DataFrame,
    calendar_df: pd.DataFrame,
    prices_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge long sales with calendar information and weekly prices.

    Filters out rows where sell_price is missing (item not yet on sale).

    Args:
        sales_long: Long format sales DataFrame.
        calendar_df: Calendar DataFrame with date, wm_yr_wk, events, snap.
        prices_df: Weekly sell prices DataFrame.

    Returns:
        pd.DataFrame merged and sorted by (item_id, store_id, date).
    """
    cal_cols = [
        "date",
        "wm_yr_wk",
        "weekday",
        "wday",
        "month",
        "year",
        "event_name_1",
        "event_type_1",
        "snap_CA",
        "snap_TX",
        "snap_WI",
        "d",
    ]
    cal_subset = calendar_df[[c for c in cal_cols if c in calendar_df.columns]].copy()
    cal_subset["d"] = cal_subset["d"].astype(str)

    # 1. Join calendar on 'd'
    merged = pd.merge(sales_long, cal_subset, on="d", how="inner")

    # Ensure match types for join with prices
    prices = prices_df.copy()
    prices["item_id"] = prices["item_id"].astype(str)
    prices["store_id"] = prices["store_id"].astype(str)
    prices["wm_yr_wk"] = prices["wm_yr_wk"].astype("int32")

    merged["item_id"] = merged["item_id"].astype(str)
    merged["store_id"] = merged["store_id"].astype(str)
    merged["wm_yr_wk"] = merged["wm_yr_wk"].astype("int32")

    # 2. Join sell prices on (item_id, store_id, wm_yr_wk)
    merged = pd.merge(
        merged,
        prices[["item_id", "store_id", "wm_yr_wk", "sell_price"]],
        on=["item_id", "store_id", "wm_yr_wk"],
        how="left",
    )

    # Missing price often indicates item was not yet on shelves; drop unpriced records
    merged = merged.dropna(subset=["sell_price"]).reset_index(drop=True)

    # Sort chronologically per series
    merged = merged.sort_values(
        by=["item_id", "store_id", "date"]
    ).reset_index(drop=True)

    return merged


def flag_outliers(
    df: pd.DataFrame,
    iqr_multiplier: float = 3.0,
) -> pd.DataFrame:
    """Flag extreme demand spikes using series-level Interquartile Range (IQR).

    Args:
        df: Merged DataFrame with 'sales', 'item_id', 'store_id'.
        iqr_multiplier: IQR distance threshold above Q3 to mark an outlier.

    Returns:
        pd.DataFrame with boolean 'is_outlier' column.
    """
    res = df.copy()

    def _get_threshold(group: pd.Series) -> float:
        q25 = group.quantile(0.25)
        q75 = group.quantile(0.75)
        iqr = q75 - q25
        return q75 + (iqr_multiplier * iqr)

    upper_bounds = res.groupby(["item_id", "store_id"])["sales"].transform(_get_threshold)
    res["is_outlier"] = (res["sales"] > upper_bounds) & (res["sales"] > 10)
    return res


def run_preprocessing(
    category: Optional[str] = DEFAULT_CATEGORY,
    stores: Optional[List[str]] = None,
    items: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Execute end-to-end loading, melting, merging, and cleaning.

    Args:
        category: Category filter.
        stores: Store IDs list.
        items: Item IDs list.

    Returns:
        Preprocessed long DataFrame with prices and calendar joined.
    """
    target_stores = stores or DEFAULT_STORES
    target_items = items or DEFAULT_ITEMS

    cal_df = load_calendar()
    prices_df = load_prices(store_ids=target_stores, item_ids=target_items)
    sales_df = load_sales(
        category=category, store_ids=target_stores, item_ids=target_items
    )

    sales_long = melt_sales_to_long(sales_df)
    clean_df = clean_and_merge(sales_long, cal_df, prices_df)
    outlier_df = flag_outliers(clean_df)

    return outlier_df
