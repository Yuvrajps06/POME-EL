"""Unit tests for Phase 6: Strategy comparison and true-world simulation framework."""

import pytest
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.experiments.simulation import MarketEnvironment
from src.experiments.comparison import (
    run_strategy_trajectory,
    run_benchmark_experiments,
    plot_cumulative_profit,
    plot_strategy_comparison_bars,
    plot_price_trajectories,
    plot_profit_distribution_box,
    plot_revenue_comparison,
    plot_sales_comparison,
    plot_improvement_distribution,
)


def test_market_environment_initialization():
    """Test MarketEnvironment initializes with correct attributes."""
    env = MarketEnvironment(
        initial_leader_price=1.38,
        initial_follower_price=1.37,
        cost=0.83,
        a=165.39,
        b=79.18,
        c=0.50,
        competitor_behavior="adaptive",
        demand_noise_std=2.0,
        competitor_noise_std=0.02,
        random_seed=42,
    )
    assert env.initial_leader_price == 1.38
    assert env.initial_follower_price == 1.37
    assert env.cost == 0.83
    assert env.period == 0
    assert env.current_follower_price == 1.37


def test_market_environment_step_and_reproducibility():
    """Test step execution produces expected schema and is strictly reproducible with fixed seed."""
    env1 = MarketEnvironment(
        initial_leader_price=1.38,
        initial_follower_price=1.37,
        cost=0.83,
        a=165.39,
        b=79.18,
        c=0.50,
        competitor_behavior="match",
        demand_noise_std=1.0,
        competitor_noise_std=0.01,
        random_seed=123,
    )
    out1 = env1.step(leader_price=1.45)

    assert "realized_follower_price" in out1
    assert "realized_demand" in out1
    assert "realized_profit" in out1
    assert "realized_revenue" in out1
    assert out1["period"] == 1
    assert out1["leader_price"] == 1.45

    # Re-instantiate with identical seed and test exact reproducibility
    env2 = MarketEnvironment(
        initial_leader_price=1.38,
        initial_follower_price=1.37,
        cost=0.83,
        a=165.39,
        b=79.18,
        c=0.50,
        competitor_behavior="match",
        demand_noise_std=1.0,
        competitor_noise_std=0.01,
        random_seed=123,
    )
    out2 = env2.step(leader_price=1.45)

    assert out1["realized_follower_price"] == out2["realized_follower_price"]
    assert out1["realized_demand"] == out2["realized_demand"]
    assert out1["realized_profit"] == out2["realized_profit"]


def test_run_strategy_trajectory():
    """Test that each strategy (A, B, C) runs for specified periods and outputs required fields."""
    env = MarketEnvironment(
        initial_leader_price=1.38,
        initial_follower_price=1.37,
        cost=0.83,
        a=165.39,
        b=79.18,
        c=0.50,
        competitor_behavior="adaptive",
        random_seed=42,
    )

    for strat in ["A", "B", "C"]:
        traj = run_strategy_trajectory(
            strategy=strat,
            env=env,
            n_periods=5,
            a=165.39,
            b=79.18,
            c=0.50,
            seed=42,
        )
        assert len(traj) == 5
        assert "cumulative_profit" in traj.columns
        assert "realized_profit" in traj.columns
        assert "recommended_price" in traj.columns
        assert "price_diff" in traj.columns
        assert not traj["realized_profit"].isna().any()
        assert not traj["recommended_price"].isna().any()


def test_common_random_numbers_demand_shocks():
    """Test that Common Random Numbers ensure identical seed yields identical noise sequence."""
    env_a = MarketEnvironment(1.38, 1.37, 0.83, 165.39, 79.18, 0.50, random_seed=999)
    env_b = MarketEnvironment(1.38, 1.37, 0.83, 165.39, 79.18, 0.50, random_seed=999)

    out_a = env_a.step(leader_price=1.45)
    out_b = env_b.step(leader_price=1.45)

    assert out_a["demand_shock"] == out_b["demand_shock"]


def test_benchmark_experiments_and_summary():
    """Test running small benchmark experiment and verify output summary structure."""
    summary_df, traj_df, runs_df = run_benchmark_experiments(
        initial_leader_price=1.38,
        initial_follower_price=1.37,
        cost=0.83,
        a=165.39,
        b=79.18,
        c=0.50,
        competitor_true_behavior="adaptive",
        n_periods=7,
        n_simulations=5,
        master_seed=42,
    )

    assert len(summary_df) == 3  # Strategies A, B, C
    assert set(summary_df["strategy"].values) == {
        "Strategy A (Baseline)",
        "Strategy B (Game Rule)",
        "Strategy C (ML-Aware)",
    }
    assert "mean_total_profit" in summary_df.columns
    assert "ci_95_lower" in summary_df.columns
    assert "ci_95_upper" in summary_df.columns
    assert "downside_risk_p5" in summary_df.columns
    assert "profit_improvement_pct" in summary_df.columns
    assert "price_volatility_std" in summary_df.columns

    assert len(runs_df) == 5
    assert len(traj_df) == 5 * 3 * 7


def test_plotting_functions():
    """Test all Phase 6 plotting functions return Plotly Figure objects."""
    summary_df, traj_df, runs_df = run_benchmark_experiments(
        initial_leader_price=1.38,
        initial_follower_price=1.37,
        cost=0.83,
        a=165.39,
        b=79.18,
        c=0.50,
        competitor_true_behavior="adaptive",
        n_periods=5,
        n_simulations=3,
        master_seed=42,
    )

    fig1 = plot_cumulative_profit(traj_df)
    fig2 = plot_strategy_comparison_bars(summary_df)
    fig3 = plot_price_trajectories(traj_df)
    fig4 = plot_profit_distribution_box(runs_df)
    fig5 = plot_revenue_comparison(traj_df)
    fig6 = plot_sales_comparison(traj_df)
    fig7 = plot_improvement_distribution(runs_df)

    for fig in [fig1, fig2, fig3, fig4, fig5, fig6, fig7]:
        assert isinstance(fig, go.Figure)
