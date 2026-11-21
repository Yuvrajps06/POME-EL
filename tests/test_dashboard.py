"""Tests for Streamlit dashboard rendering and integration edge cases."""

import pytest
import pandas as pd
from unittest.mock import patch, MagicMock

from dashboard import (
    overview,
    market,
    demand,
    competitor,
    optimizer,
    comparison,
    sensitivity,
    utils,
)
from src.optimization.baseline_optimizer import optimize_baseline
from src.optimization.game_optimizer import optimize_game_rule
from src.optimization.ml_optimizer import optimize_ml_stackelberg


def test_dashboard_modules_importable():
    """Verify all dashboard pages are importable and contain a callable render function."""
    pages = [overview, market, demand, competitor, optimizer, comparison, sensitivity]
    for page in pages:
        assert hasattr(page, "render")
        assert callable(page.render)


def test_dashboard_utils_caching_and_graceful_missing(tmp_path):
    """Test dashboard utilities handle missing files gracefully without crashing."""
    try:
        utils.load_processed_dataset.clear()
        utils.load_strategy_comparison_data.clear()
        utils.load_sensitivity_datasets.clear()
        with patch("dashboard.utils.DATA_PROCESSED_DIR", tmp_path):
            df = utils.load_processed_dataset()
            assert df.empty

            d1, d2, d3 = utils.load_strategy_comparison_data()
            assert d1.empty and d2.empty and d3.empty

            sens_dict = utils.load_sensitivity_datasets()
            assert all(v.empty for v in sens_dict.values())
    finally:
        utils.load_processed_dataset.clear()
        utils.load_strategy_comparison_data.clear()
        utils.load_sensitivity_datasets.clear()


def test_optimizer_edge_cases_extreme_grid():
    """Test optimizer edge cases: very wide grid and zero margin conditions."""
    # Extreme search grid: 20% to 250% of current price
    res = optimize_baseline(
        baseline_price=1.38,
        current_follower_price=1.37,
        cost=0.83,
        a=165.39,
        b=79.18,
        c=0.50,
        min_factor=0.20,
        max_factor=2.50,
        num_steps=101,
    )
    assert res.recommended_price > 0
    assert not res.boundary_warning  # Inside wide bounds
    assert not res.negative_margin


def test_optimizer_edge_case_cost_exceeds_price():
    """Test optimizer edge case when cost exceeds baseline price."""
    # When cost = $2.00 > $1.38
    res = optimize_baseline(
        baseline_price=1.38,
        current_follower_price=1.37,
        cost=2.00,
        a=165.39,
        b=79.18,
        c=0.50,
    )
    # Recommended price should adjust upward to preserve margin or flag negative margin
    assert res.recommended_price >= res.grid_min
    if res.recommended_price < 2.00:
        assert res.negative_margin is True
