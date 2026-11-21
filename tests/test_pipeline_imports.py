"""Smoke test verifying all imports and entry points used by scripts/run_pipeline.py."""

import pytest


def test_data_loader_imports() -> None:
    """Verify data loader public functions."""
    from src.data.loader import (
        load_calendar,
        load_prices,
        load_sales,
        rank_candidate_items,
    )

    assert callable(load_calendar)
    assert callable(load_prices)
    assert callable(load_sales)
    assert callable(rank_candidate_items)


def test_preprocessing_imports() -> None:
    """Verify data preprocessing public functions."""
    from src.data.preprocessing import (
        clean_and_merge,
        flag_outliers,
        melt_sales_to_long,
        run_preprocessing,
    )

    assert callable(clean_and_merge)
    assert callable(flag_outliers)
    assert callable(melt_sales_to_long)
    assert callable(run_preprocessing)


def test_feature_engineering_imports() -> None:
    """Verify feature engineering public functions."""
    from src.data.feature_engineering import (
        add_calendar_features,
        add_lag_features,
        add_log_transforms,
        add_price_features,
        add_rolling_features,
        build_feature_pipeline,
        chronological_split,
    )

    assert callable(add_calendar_features)
    assert callable(add_lag_features)
    assert callable(add_log_transforms)
    assert callable(add_price_features)
    assert callable(add_rolling_features)
    assert callable(build_feature_pipeline)
    assert callable(chronological_split)


def test_config_imports() -> None:
    """Verify config variables used by the pipeline."""
    from src.utils.config import (
        DATA_PROCESSED_DIR,
        DATA_RAW_DIR,
        DATA_SAMPLE_DIR,
        DEFAULT_CATEGORY,
        DEFAULT_ITEMS,
        DEFAULT_STORES,
        SELECTED_ITEMS,
        SELECTED_STORES,
    )

    assert DATA_RAW_DIR is not None
    assert DATA_PROCESSED_DIR is not None
    assert DATA_SAMPLE_DIR is not None
    assert isinstance(DEFAULT_CATEGORY, str)
    assert len(DEFAULT_STORES) > 0
    assert len(DEFAULT_ITEMS) > 0
    assert SELECTED_ITEMS == DEFAULT_ITEMS
    assert SELECTED_STORES == DEFAULT_STORES


def test_notebook_runner_imports() -> None:
    """Verify notebook execution entry points invoked by run_pipeline."""
    from notebooks.run_competitor_modeling import main as run_competitor
    from notebooks.run_demand_modeling import main as run_demand
    from notebooks.run_eda import main as run_eda
    from notebooks.run_sensitivity_analysis import main as run_sensitivity
    from notebooks.run_strategy_comparison import main as run_comparison

    assert callable(run_competitor)
    assert callable(run_demand)
    assert callable(run_eda)
    assert callable(run_sensitivity)
    assert callable(run_comparison)


def test_run_pipeline_callable() -> None:
    """Verify run_pipeline script function import."""
    from scripts.run_pipeline import run_pipeline

    assert callable(run_pipeline)
