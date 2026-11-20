"""Strategy C: ML-Aware Stackelberg pricing optimizer.

Optimizes leader price anticipating a follower whose price response is predicted
by the trained ML competitor response model:
    P_L* = argmax (P_L - C_L) * Q_L(P_L, R_ML(P_L, X))
Evaluates over a bounded candidate price grid, supports noisy ML degradation
for sensitivity analysis, and flags boundary or clipping edge cases.
"""

from typing import Any, Dict, Optional
import numpy as np
import pandas as pd

from src.competitor.response_model import CompetitorResponseModel, get_competitor_price
from src.game.demand_function import linear_demand
from src.game.profit_function import calculate_margin, calculate_margin_pct, calculate_profit
from src.optimization.baseline_optimizer import OptimizationResult
from src.utils.config import (
    PRICE_GRID_MAX_FACTOR,
    PRICE_GRID_MIN_FACTOR,
    PRICE_GRID_NUM_STEPS,
)


def optimize_ml_stackelberg(
    baseline_price: float,
    current_follower_price: float,
    cost: float,
    a: float,
    b: float,
    c: float = 0.50,
    context: Optional[Dict[str, Any]] = None,
    context_shift: float = 0.0,
    competitor_model: Optional[CompetitorResponseModel] = None,
    target_accuracy: Optional[float] = None,
    min_factor: float = PRICE_GRID_MIN_FACTOR,
    max_factor: float = PRICE_GRID_MAX_FACTOR,
    num_steps: int = PRICE_GRID_NUM_STEPS,
    custom_grid: Optional[np.ndarray] = None,
) -> OptimizationResult:
    """Optimize price under Strategy C: ML-Aware Stackelberg with ML-predicted follower response.

    Args:
        baseline_price: Current firm price to center candidate grid.
        current_follower_price: Prevailing competitor follower price.
        cost: Exogenous unit cost.
        a: Base demand intercept.
        b: Own-price sensitivity slope.
        c: Cross-price sensitivity slope.
        context: Environmental covariates dictionary (demand, calendar, SNAP).
        context_shift: Exogenous demand shift (d * X).
        competitor_model: Optional pre-loaded CompetitorResponseModel.
        target_accuracy: Optional target accuracy in [0.50, 1.0] for noisy degradation.
        min_factor: Lower grid bound factor.
        max_factor: Upper grid bound factor.
        num_steps: Grid resolution.
        custom_grid: Optional explicit price array.

    Returns:
        OptimizationResult object with optimal recommendation, ML-predicted follower price, and grid.
    """
    p_base = max(0.10, float(baseline_price))
    c_unit = float(cost)
    p_foll = float(current_follower_price)
    ctx = context or {}

    # ML competitor parameters
    ml_params: Dict[str, Any] = {}
    if competitor_model is not None:
        ml_params["model"] = competitor_model
    if target_accuracy is not None:
        ml_params["target_accuracy"] = float(target_accuracy)

    # 1. Define candidate price grid
    if custom_grid is not None:
        grid = np.sort(np.asarray(custom_grid, dtype=float))
    else:
        grid_min = max(0.10, p_base * float(min_factor))
        grid_max = p_base * float(max_factor)
        grid = np.linspace(grid_min, grid_max, int(num_steps))

    # 2. Evaluate grid anticipating ML-predicted follower response
    model_obj = competitor_model or ml_params.get("model")
    if model_obj is None and target_accuracy is None:
        from src.competitor.response_model import load_competitor_model
        try:
            model_obj = load_competitor_model()
        except Exception:
            model_obj = None

    if target_accuracy is None and model_obj is not None and hasattr(model_obj, "predict_response"):
        # Fast vectorized path
        leader_prev = float(ctx.get("leader_prev_price", p_foll))
        leader_pct = (grid - leader_prev) / max(0.01, leader_prev)
        df_grid = pd.DataFrame({
            "leader_price": grid,
            "leader_prev_price": leader_prev,
            "leader_price_change_pct": leader_pct,
            "follower_price": p_foll,
            "price_gap": grid - p_foll,
            "price_ratio": grid / max(0.01, p_foll),
            "recent_demand": float(ctx.get("recent_demand", 10.0)),
            "sales_volatility": float(ctx.get("sales_volatility", 2.0)),
            "demand_trend": float(ctx.get("demand_trend", 0.0)),
            "day_of_week": int(ctx.get("day_of_week", 2)),
            "month": int(ctx.get("month", 5)),
            "is_weekend": int(ctx.get("is_weekend", 0)),
            "snap_CA": int(ctx.get("snap_CA", 0)),
            "has_event": int(ctx.get("has_event", 0)),
        })
        follower_prices, prob_increases = model_obj.predict_response(df_grid)
        demands = np.maximum(0.0, a - b * grid + c * follower_prices + context_shift)
        profits = (grid - c_unit) * demands
        margins = (grid - c_unit) / np.maximum(0.01, grid)
        revenues = grid * demands

        best_idx = int(np.argmax(profits))
        grid_df = pd.DataFrame({
            "price": np.round(grid, 4),
            "follower_price": np.round(follower_prices, 4),
            "prob_increase": np.round(prob_increases, 4),
            "demand": np.round(demands, 4),
            "profit": np.round(profits, 4),
            "margin": np.round(margins, 4),
            "revenue": np.round(revenues, 4),
        })
    else:
        # Standard loop path (e.g. for noisy wrapper or non-standard models)
        records = []
        best_profit = -np.inf
        best_idx = 0

        for i, p in enumerate(grid):
            # Follower responds via ML predictor
            follower_new_p, prob_inc = get_competitor_price(
                leader_price=p,
                current_follower_price=p_foll,
                context=ctx,
                mode="simulated_ml",
                params=ml_params,
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

    strat_label = (
        f"ML-Aware Stackelberg ({int(target_accuracy * 100)}% Acc)"
        if target_accuracy is not None
        else "ML-Aware Stackelberg"
    )

    return OptimizationResult(
        strategy=strat_label,
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
