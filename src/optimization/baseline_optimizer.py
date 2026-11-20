"""Strategy A: Baseline (Traditional) pricing optimizer.

Maximizes firm profit ignoring competitor reactions, assuming competitor price
remains static at current levels:
    P_L* = argmax (P_L - C_L) * Q_L(P_L, P_F_fixed)
Evaluates over a bounded candidate price grid and detects boundary or clipping edge cases.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.game.demand_function import linear_demand
from src.game.profit_function import calculate_margin, calculate_margin_pct, calculate_profit
from src.utils.config import (
    PRICE_GRID_MAX_FACTOR,
    PRICE_GRID_MIN_FACTOR,
    PRICE_GRID_NUM_STEPS,
)


@dataclass
class OptimizationResult:
    """Standardized output container for pricing strategy optimizations."""

    strategy: str
    recommended_price: float
    predicted_follower_price: float
    expected_demand: float
    expected_profit: float
    expected_revenue: float
    cost: float
    unit_margin: float
    margin_pct: float
    baseline_price: float
    grid_min: float
    grid_max: float
    at_min_boundary: bool
    at_max_boundary: bool
    boundary_warning: bool
    q_clipped: bool
    negative_margin: bool
    grid_evaluations: pd.DataFrame

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary format (excluding DataFrame for compact serialization)."""
        return {
            "strategy": self.strategy,
            "recommended_price": self.recommended_price,
            "predicted_follower_price": self.predicted_follower_price,
            "expected_demand": self.expected_demand,
            "expected_profit": self.expected_profit,
            "expected_revenue": self.expected_revenue,
            "cost": self.cost,
            "unit_margin": self.unit_margin,
            "margin_pct": self.margin_pct,
            "baseline_price": self.baseline_price,
            "grid_min": self.grid_min,
            "grid_max": self.grid_max,
            "at_min_boundary": self.at_min_boundary,
            "at_max_boundary": self.at_max_boundary,
            "boundary_warning": self.boundary_warning,
            "q_clipped": self.q_clipped,
            "negative_margin": self.negative_margin,
        }


def optimize_baseline(
    baseline_price: float,
    current_follower_price: float,
    cost: float,
    a: float,
    b: float,
    c: float = 0.50,
    context_shift: float = 0.0,
    min_factor: float = PRICE_GRID_MIN_FACTOR,
    max_factor: float = PRICE_GRID_MAX_FACTOR,
    num_steps: int = PRICE_GRID_NUM_STEPS,
    custom_grid: Optional[np.ndarray] = None,
) -> OptimizationResult:
    """Optimize price under Strategy A: Traditional monopoly demand ignoring competitor feedback.

    Args:
        baseline_price: Current firm price to center candidate grid.
        current_follower_price: Competitor price (held fixed).
        cost: Exogenous unit cost parameter.
        a: Base demand intercept.
        b: Own-price sensitivity slope.
        c: Cross-price sensitivity slope.
        context_shift: Exogenous demand shifter (d * X).
        min_factor: Lower grid bound as factor of baseline (default: 0.70).
        max_factor: Upper grid bound as factor of baseline (default: 1.30).
        num_steps: Discrete evaluation points count.
        custom_grid: Optional explicit price array.

    Returns:
        OptimizationResult object with optimal recommendation and full evaluation grid.
    """
    p_base = max(0.10, float(baseline_price))
    c_unit = float(cost)
    p_foll = float(current_follower_price)

    # 1. Define candidate price grid
    if custom_grid is not None:
        grid = np.sort(np.asarray(custom_grid, dtype=float))
    else:
        grid_min = max(0.10, p_base * float(min_factor))
        grid_max = p_base * float(max_factor)
        grid = np.linspace(grid_min, grid_max, int(num_steps))

    # 2. Evaluate grid
    records = []
    best_profit = -np.inf
    best_idx = 0

    for i, p in enumerate(grid):
        # Follower price remains fixed under baseline assumption
        q = linear_demand(
            leader_price=p,
            follower_price=p_foll,
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
            "follower_price": round(float(p_foll), 4),
            "demand": round(float(q), 4),
            "profit": round(float(profit), 4),
            "margin": round(float(margin), 4),
            "revenue": round(float(p * q), 4),
        })

    grid_df = pd.DataFrame(records)

    # 3. Extract optimal recommendation
    best_row = grid_df.iloc[best_idx]
    opt_p = float(best_row["price"])
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
        strategy="Traditional (Baseline)",
        recommended_price=round(opt_p, 2),
        predicted_follower_price=round(p_foll, 2),
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


def plot_profit_vs_price(
    results: Union[OptimizationResult, List[OptimizationResult]],
) -> go.Figure:
    """Plot profit curves across candidate price grid for one or multiple optimization results.

    Args:
        results: Single OptimizationResult or list of OptimizationResult objects.

    Returns:
        Plotly Figure.
    """
    if isinstance(results, OptimizationResult):
        results = [results]

    fig = go.Figure()

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    for i, r in enumerate(results):
        df = r.grid_evaluations
        c = colors[i % len(colors)]
        fig.add_trace(
            go.Scatter(
                x=df["price"],
                y=df["profit"],
                mode="lines",
                name=f"{r.strategy} Profit Curve",
                line=dict(color=c, width=2.5),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[r.recommended_price],
                y=[r.expected_profit],
                mode="markers",
                marker=dict(size=12, symbol="star", color=c),
                name=f"{r.strategy} Optimum (${r.recommended_price:.2f})",
            )
        )

    fig.update_layout(
        title="Expected Profit vs Leader Candidate Price ($)",
        xaxis_title="Leader Price ($)",
        yaxis_title="Expected Profit ($)",
        template="plotly_white",
        hovermode="x unified",
    )
    return fig

