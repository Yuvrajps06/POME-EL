"""Tests for Phase 5 pricing optimization and Stackelberg game engine."""

import numpy as np
import pandas as pd
import pytest

from src.game.demand_function import calculate_context_shift, linear_demand
from src.game.profit_function import (
    calculate_margin,
    calculate_margin_pct,
    calculate_profit,
    summarize_financials,
)
from src.game.stackelberg import (
    closed_form_baseline_monopoly,
    closed_form_stackelberg,
)
from src.optimization.baseline_optimizer import (
    OptimizationResult,
    optimize_baseline,
)
from src.optimization.game_optimizer import optimize_game_rule
from src.optimization.ml_optimizer import optimize_ml_stackelberg


def test_linear_demand_and_clipping() -> None:
    """Verify linear demand calculation and non-negative quantity clipping."""
    # Q = 100 - 10*P_L + 2*P_F + 0
    # At P_L = 5, P_F = 5: Q = 100 - 50 + 10 = 60
    q = linear_demand(leader_price=5.0, follower_price=5.0, a=100.0, b=10.0, c=2.0)
    assert q == 60.0

    # At extreme price P_L = 20: Q = 100 - 200 + 10 = -90 -> clipped to 0.0
    q_choked = linear_demand(leader_price=20.0, follower_price=5.0, a=100.0, b=10.0, c=2.0)
    assert q_choked == 0.0


def test_profit_and_negative_margin() -> None:
    """Verify profit calculation and negative margin handling."""
    # P = 5.0, C = 3.0, Q = 10 -> Profit = (5 - 3) * 10 = 20
    assert calculate_profit(price=5.0, cost=3.0, demand=10.0) == 20.0

    # Negative margin: P = 2.0, C = 3.0, Q = 10 -> Profit = (2 - 3) * 10 = -10
    assert calculate_profit(price=2.0, cost=3.0, demand=10.0) == -10.0
    assert calculate_margin(price=2.0, cost=3.0) == -1.0
    assert calculate_margin_pct(price=2.0, cost=3.0) == -0.50

    # Zero demand yields zero profit even with negative margin
    assert calculate_profit(price=2.0, cost=3.0, demand=0.0) == 0.0


def test_known_answer_closed_form_vs_grid() -> None:
    """Verify that numerical grid optimizer matches closed-form analytical optimum."""
    # Specification:
    # Q_L = 100 - 10*P_L + 2*P_F
    # Follower: P_F = 1.0 + 0.5*P_L
    # Leader cost: C_L = 2.0
    # Net demand: Q_L = (100 + 2*1.0) - (10 - 2*0.5)*P_L = 102 - 9*P_L
    # Analytical: P_L* = (102 + 9*2.0) / (2 * 9) = 120 / 18 = 6.6667
    a, b, c = 100.0, 10.0, 2.0
    alpha, beta = 1.0, 0.5
    cost = 2.0

    cf = closed_form_stackelberg(a=a, b=b, c=c, alpha=alpha, beta=beta, cost=cost)
    assert pytest.approx(cf["optimal_price"], 0.001) == 6.6667
    assert pytest.approx(cf["expected_demand"], 0.01) == 42.0
    assert pytest.approx(cf["expected_profit"], 0.01) == 196.0

    # Test fine numerical grid optimizer
    fine_grid = np.linspace(4.0, 9.0, 1001)
    # Using baseline optimizer with effective parameters
    res = optimize_baseline(
        baseline_price=6.0,
        current_follower_price=cf["follower_price"],
        cost=cost,
        a=102.0,
        b=9.0,
        c=0.0,
        custom_grid=fine_grid,
    )
    assert pytest.approx(res.recommended_price, 0.01) == 6.67
    assert pytest.approx(res.expected_profit, 0.1) == 196.0


def test_stackelberg_equals_baseline_when_competitor_ignored() -> None:
    """Verify that Stackelberg reduces to Baseline when c = 0 or follower is static."""
    a, b = 100.0, 10.0
    cost = 2.0
    follower_p = 5.0
    grid = np.linspace(4.0, 8.0, 41)

    # When c = 0, competitor reaction has zero influence on leader demand
    res_base = optimize_baseline(
        baseline_price=6.0,
        current_follower_price=follower_p,
        cost=cost,
        a=a,
        b=b,
        c=0.0,
        custom_grid=grid,
    )

    res_game = optimize_game_rule(
        baseline_price=6.0,
        current_follower_price=follower_p,
        cost=cost,
        a=a,
        b=b,
        c=0.0,
        rule_params={"behavior": "match"},
        custom_grid=grid,
    )

    assert res_base.recommended_price == res_game.recommended_price
    assert res_base.expected_profit == res_game.expected_profit
    assert res_base.expected_demand == res_game.expected_demand


def test_cost_monotonicity() -> None:
    """Verify that higher unit cost leads to higher (or equal) optimal prices."""
    a, b, c = 120.0, 15.0, 1.0
    p_base = 5.0
    p_foll = 5.0

    res_low_cost = optimize_baseline(p_base, p_foll, cost=1.0, a=a, b=b, c=c)
    res_high_cost = optimize_baseline(p_base, p_foll, cost=3.0, a=a, b=b, c=c)

    assert res_high_cost.recommended_price >= res_low_cost.recommended_price


def test_grid_boundary_edge_cases() -> None:
    """Verify boundary warning flags when optimum is constrained by grid edges."""
    a, b, c = 100.0, 5.0, 0.0
    cost = 2.0
    # Unconstrained monopoly price: P* = 2/2 + 100 / (2*5) = 1 + 10 = 11.0

    # Grid constrained below optimum: [4.0, 8.0]
    res_upper = optimize_baseline(
        baseline_price=6.0,
        current_follower_price=5.0,
        cost=cost,
        a=a,
        b=b,
        c=c,
        custom_grid=np.linspace(4.0, 8.0, 9),
    )
    assert res_upper.recommended_price == 8.0
    assert res_upper.at_max_boundary is True
    assert res_upper.boundary_warning is True

    # Grid constrained above optimum: [12.0, 15.0]
    res_lower = optimize_baseline(
        baseline_price=13.0,
        current_follower_price=5.0,
        cost=cost,
        a=a,
        b=b,
        c=c,
        custom_grid=np.linspace(12.0, 15.0, 7),
    )
    assert res_lower.recommended_price == 12.0
    assert res_lower.at_min_boundary is True
    assert res_lower.boundary_warning is True


def test_all_three_strategies_run_cleanly() -> None:
    """Verify Baseline, Game-Theoretic, and ML-Aware optimizers execute consistently."""
    base_p = 1.38
    foll_p = 1.37
    cost = 0.83
    a, b, c = 165.39, 79.18, 0.50

    res_a = optimize_baseline(base_p, foll_p, cost, a, b, c)
    res_b = optimize_game_rule(base_p, foll_p, cost, a, b, c, rule_params={"behavior": "match"})
    res_c = optimize_ml_stackelberg(base_p, foll_p, cost, a, b, c)

    for res in [res_a, res_b, res_c]:
        assert isinstance(res, OptimizationResult)
        assert res.recommended_price > 0.0
        assert res.expected_demand > 0.0
        assert res.expected_profit > 0.0
        assert not res.grid_evaluations.empty
        assert "price" in res.grid_evaluations.columns
        assert "profit" in res.grid_evaluations.columns
