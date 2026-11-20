"""Unit tests for Phase 7: Sensitivity analysis & simulation experiments."""

from pathlib import Path
import pytest
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.competitor.response_model import CompetitorResponseModel, load_competitor_model
from src.experiments.sensitivity import (
    sweep_accuracy_and_response_strength,
    sweep_accuracy_curves,
    sweep_competitor_behaviors,
    sweep_demand_sensitivity,
    sweep_robustness_uncertainty,
    plot_heatmap_strength_x_accuracy,
    plot_accuracy_vs_profit_curve,
    plot_competitor_behavior_comparison,
    plot_demand_sensitivity_vs_price,
    plot_robustness_analysis,
)


def test_sweep_accuracy_and_response_strength(tmp_path):
    """Test response strength x accuracy sweep produces expected schema and caching works."""
    cache_file = tmp_path / "test_heatmap.csv"

    df = sweep_accuracy_and_response_strength(
        n_simulations=2,
        n_periods=3,
        cache_path=cache_file,
        force_recompute=True,
    )

    assert not df.empty
    assert "response_strength" in df.columns
    assert "target_accuracy" in df.columns
    assert "profit_improvement_pct" in df.columns
    assert cache_file.exists()

    # Test cache read
    df_cached = sweep_accuracy_and_response_strength(
        n_simulations=2,
        n_periods=3,
        cache_path=cache_file,
        force_recompute=False,
    )
    assert len(df_cached) == len(df)


def test_sweep_accuracy_curves_asymmetric_errors(tmp_path):
    """Test accuracy sweep across symmetric, optimistic, and pessimistic error regimes."""
    cache_file = tmp_path / "test_acc_curves.csv"

    df = sweep_accuracy_curves(
        n_simulations=2,
        n_periods=3,
        cache_path=cache_file,
        force_recompute=True,
    )

    assert not df.empty
    assert set(df["error_mode"].unique()) == {"symmetric", "optimistic", "pessimistic"}
    assert "c_wins_pct" in df.columns
    assert "profit_improvement_pct" in df.columns


def test_sweep_competitor_behaviors_and_delays(tmp_path):
    """Test sweep across behaviors (match, undercut, maintain, raise) and delay periods (0, 1, 3)."""
    cache_file = tmp_path / "test_behaviors.csv"

    df = sweep_competitor_behaviors(
        n_simulations=2,
        n_periods=4,
        cache_path=cache_file,
        force_recompute=True,
    )

    assert not df.empty
    assert set(df["competitor_behavior"].unique()) == {"match", "undercut", "maintain", "raise"}
    assert set(df["delay_periods"].unique()) == {0, 1, 3}
    assert "improvement_C_over_A_pct" in df.columns


def test_sweep_demand_sensitivity(tmp_path):
    """Test demand elasticity sweep across own-price b and cross-price c."""
    cache_file = tmp_path / "test_demand_slopes.csv"

    df = sweep_demand_sensitivity(
        cache_path=cache_file,
        force_recompute=True,
    )

    assert not df.empty
    assert "own_price_b" in df.columns
    assert "cross_price_c" in df.columns
    assert "price_C" in df.columns
    assert "price_A" in df.columns


def test_sweep_robustness_uncertainty(tmp_path):
    """Test robustness sweep under cost levels and unobserved demand volatility."""
    cache_file = tmp_path / "test_robustness.csv"

    df = sweep_robustness_uncertainty(
        n_simulations=2,
        n_periods=3,
        cache_path=cache_file,
        force_recompute=True,
    )

    assert not df.empty
    assert "unit_cost" in df.columns
    assert "demand_volatility" in df.columns
    assert "profit_difference" in df.columns


def test_plotly_figures_generation(tmp_path):
    """Test all Plotly figure functions return valid Figure objects."""
    df_heat = sweep_accuracy_and_response_strength(n_simulations=2, n_periods=3, cache_path=tmp_path / "h.csv")
    df_acc = sweep_accuracy_curves(n_simulations=2, n_periods=3, cache_path=tmp_path / "a.csv")
    df_beh = sweep_competitor_behaviors(n_simulations=2, n_periods=4, cache_path=tmp_path / "b.csv")
    df_dem = sweep_demand_sensitivity(cache_path=tmp_path / "d.csv")
    df_rob = sweep_robustness_uncertainty(n_simulations=2, n_periods=3, cache_path=tmp_path / "r.csv")

    fig1 = plot_heatmap_strength_x_accuracy(df_heat)
    fig2 = plot_accuracy_vs_profit_curve(df_acc)
    fig3 = plot_competitor_behavior_comparison(df_beh)
    fig4 = plot_demand_sensitivity_vs_price(df_dem)
    fig5 = plot_robustness_analysis(df_rob)

    for fig in [fig1, fig2, fig3, fig4, fig5]:
        assert isinstance(fig, go.Figure)
