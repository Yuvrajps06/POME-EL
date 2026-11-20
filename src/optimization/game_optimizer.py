"""Strategy B: Game-Theoretic (Rule-Based Stackelberg) pricing optimizer.

Optimizes leader price anticipating a follower who responds according to an
assumed rule-based reaction policy (match, undercut, maintain, raise, or adaptive):
    P_L* = argmax (P_L - C_L) * Q_L(P_L, R_rule(P_L))
Evaluates over a bounded candidate price grid and detects boundary or clipping edge cases.
"""

from typing import Any, Dict, Optional
import numpy as np
import pandas as pd

from src.competitor.response_model import get_competitor_price
from src.game.demand_function import linear_demand
from src.game.profit_function import calculate_margin, calculate_margin_pct, calculate_profit
from src.optimization.baseline_optimizer import OptimizationResult
from src.utils.config import (
    PRICE_GRID_MAX_FACTOR,
    PRICE_GRID_MIN_FACTOR,
    PRICE_GRID_NUM_STEPS,
)


def optimize_game_rule(
    baseline_price: float,
    current_follower_price: float,
    cost: float,
    a: float,
    b: float,
    c: float = 0.50,
    context_shift: float = 0.0,
    rule_params: Optional[Dict[str, Any]] = None,
    min_factor: float = PRICE_GRID_MIN_FACTOR,
    max_factor: float = PRICE_GRID_MAX_FACTOR,
    num_steps: int = PRICE_GRID_NUM_STEPS,
    custom_grid: Optional[np.ndarray] = None,
) -> OptimizationResult:
    """Optimize price under Strategy B: Stackelberg with rule-based follower response.

    Args:
        baseline_price: Current firm price to center candidate grid.
        current_follower_price: Prevailing competitor follower price.
        cost: Exogenous unit cost.
        a: Base demand intercept.
        b: Own-price sensitivity slope.
        c: Cross-price sensitivity slope.
        context_shift: Exogenous demand shift (d * X).
        rule_params: Follower rule parameters (behavior, match_probability, undercut_margin).
        min_factor: Lower grid bound factor.
        max_factor: Upper grid bound factor.
        num_steps: Grid resolution.
        custom_grid: Optional explicit price array.

    Returns:
        OptimizationResult object with optimal recommendation, expected follower price, and grid.
    """
    p_base = max(0.10, float(baseline_price))
    c_unit = float(cost)
    p_foll = float(current_follower_price)
    r_params = rule_params or {"behavior": "match", "match_probability": 0.85}

    # 1. Define candidate price grid
    if custom_grid is not None:
        grid = np.sort(np.asarray(custom_grid, dtype=float))
    else:
        grid_min = max(0.10, p_base * float(min_factor))
        grid_max = p_base * float(max_factor)
        grid = np.linspace(grid_min, grid_max, int(num_steps))

    # 2. Evaluate grid anticipating follower's rule-based response
    records = []
    best_profit = -np.inf
    best_idx = 0

    for i, p in enumerate(grid):
        # Follower responds via configured behavioral rule
        follower_new_p, prob_inc = get_competitor_price(
            leader_price=p,
            current_follower_price=p_foll,
            mode="rule",
            params=r_params,
        )

        q = linear_demand(
            leader_price=p,
            follower_price=follower_new_p,
            a=a,
            b=b,
            c=c,
            context_shift=context_shift,
        )
        profit = calculate_profit(p, c_unit, q)
        margin = calculate_margin(p, c_unit)

        if profit > best_profit:
            best_profit = profit
            best_idx = i

        records.append({
            "price": round(float(p), 4),
            "follower_price": round(float(follower_new_p), 4),
            "prob_increase": round(float(prob_inc), 4),
            "demand": round(float(q), 4),
            "profit": round(float(profit), 4),
            "margin": round(float(margin), 4),
            "revenue": round(float(p * q), 4),
        })

    grid_df = pd.DataFrame(records)

    # 3. Extract optimal recommendation
    best_row = grid_df.iloc[best_idx]
    opt_p = float(best_row["price"])
    opt_pf = float(best_row["follower_price"])
    opt_q = float(best_row["demand"])
    opt_prof = float(best_row["profit"])
    opt_rev = float(best_row["revenue"])
    opt_margin = float(best_row["margin"])

    # 4. Check edge cases
    at_min = bool(best_idx == 0)
    at_max = bool(best_idx == len(grid) - 1)
    boundary_warn = bool(at_min or at_max)
    is_q_clipped = bool(opt_q <= 0.0)
    is_neg_margin = bool(opt_margin < 0.0)

    return OptimizationResult(
        strategy="Game-Theoretic (Rule-Based Stackelberg)",
        recommended_price=round(opt_p, 2),
        predicted_follower_price=round(opt_pf, 2),
        expected_demand=round(opt_q, 2),
        expected_profit=round(opt_prof, 2),
        expected_revenue=round(opt_rev, 2),
        cost=round(c_unit, 2),
        unit_margin=round(opt_margin, 2),
        margin_pct=round(calculate_margin_pct(opt_p, c_unit), 4),
        baseline_price=round(p_base, 2),
        grid_min=round(float(grid[0]), 2),
        grid_max=round(float(grid[-1]), 2),
        at_min_boundary=at_min,
        at_max_boundary=at_max,
        boundary_warning=boundary_warn,
        q_clipped=is_q_clipped,
        negative_margin=is_neg_margin,
        grid_evaluations=grid_df,
    )
