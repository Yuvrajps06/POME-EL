"""Competitor response simulation module.

Simulates strategic competitor (follower) reaction logic under explicit behavioral
rules. Generates synthetic follower training datasets from known behavioral rules
with noise for subsequent ML modeling and rule-recovery benchmarking.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from src.utils.config import RANDOM_SEED

SIMULATION_DISCLAIMER: str = (
    "DATA PROVENANCE NOTICE: M5 retail data contains NO external competitor prices. "
    "All competitor reaction models in mode 'simulated_ml' and 'rule' are derived from "
    "calibrated behavioral simulations with known response rules. Never claim competitor "
    "behavior was learned from M5 retail data directly."
)


def rule_based_response(
    leader_price: float,
    current_follower_price: float,
    context: Optional[Dict[str, Any]] = None,
    params: Optional[Dict[str, Any]] = None,
) -> Tuple[float, float]:
    """Compute rule-based competitor response to leader's pricing decision.

    Behaviors supported:
    - 'match': Matches leader price with probability match_probability.
    - 'undercut': Sets price at (1 - margin) * leader_price.
    - 'maintain': Maintains current price unless leader undercuts drastically.
    - 'raise': Follows leader upwards or establishes premium.
    - 'adaptive': Blended reaction based on leader price change and price gap.

    Args:
        leader_price: Leader's proposed or current price.
        current_follower_price: Follower's existing price.
        context: Optional dictionary with environment features (demand, trend).
        params: Configuration parameters:
            - behavior: 'match', 'undercut', 'maintain', 'raise', 'adaptive' (default: 'adaptive')
            - match_probability: Float in [0, 1] (default: 0.85)
            - response_strength: Elasticity of reaction (default: 1.0)
            - undercut_margin: Margin below leader (default: 0.03 = 3%)
            - raise_margin: Margin above leader (default: 0.04 = 4%)
            - noise_std: Standard deviation of additive price noise (default: 0.0)

    Returns:
        Tuple of (follower_price: float, probability_of_increase: float).
    """
    p = params or {}
    behavior = str(p.get("behavior", "adaptive")).lower()
    match_prob = float(p.get("match_probability", 0.85))
    resp_strength = float(p.get("response_strength", 1.0))
    undercut_margin = float(p.get("undercut_margin", 0.03))
    raise_margin = float(p.get("raise_margin", 0.04))
    noise_std = float(p.get("noise_std", 0.0))

    p_leader = max(0.01, float(leader_price))
    p_follower = max(0.01, float(current_follower_price))

    price_diff = p_leader - p_follower
    leader_pct_change = (p_leader - p_follower) / p_follower

    if behavior == "match":
        # Follower matches leader price with match_prob
        target_price = p_leader
        prob_increase = 0.95 if p_leader > p_follower else 0.05

    elif behavior == "undercut":
        # Always undercut leader by undercut_margin
        target_price = p_leader * (1.0 - undercut_margin)
        prob_increase = 0.85 if target_price > p_follower else 0.10

    elif behavior == "raise":
        # Price at a premium or raise along with leader
        target_price = max(p_leader * (1.0 + raise_margin), p_follower * 1.02)
        prob_increase = 0.95

    elif behavior == "maintain":
        # Maintain price unless leader significantly undercuts (> 10%)
        if leader_pct_change < -0.10:
            target_price = p_leader * (1.0 - undercut_margin)
            prob_increase = 0.05
        else:
            target_price = p_follower
            prob_increase = 0.50

    elif behavior == "adaptive":
        # Adaptive response: blends price gap, demand velocity, and leader direction
        # Sigmoid probability of follower increasing price:
        z = 4.0 * leader_pct_change + 2.0 * (price_diff / p_follower)
        prob_increase = float(1.0 / (1.0 + np.exp(-z)))

        # Magnitude reaction
        if leader_pct_change > 0:
            # Leader raised price -> follower follows partially by resp_strength
            target_price = p_follower * (1.0 + resp_strength * leader_pct_change * 0.75)
        else:
            # Leader lowered price -> follower matches aggressively to protect share
            target_price = p_follower * (1.0 + resp_strength * leader_pct_change * 0.90)

    else:
        raise ValueError(
            f"Unknown behavior '{behavior}'. Supported: 'match', 'undercut', 'maintain', 'raise', 'adaptive'."
        )

    # Apply noise if configured
    if noise_std > 0.0:
        noise = np.random.normal(0.0, noise_std * target_price)
        final_price = max(0.10, target_price + noise)
    else:
        final_price = max(0.10, target_price)

    # Recalculate true binary direction probability for precision
    if final_price > (p_follower + 1e-4):
        prob_increase = max(prob_increase, 0.75)
    elif final_price < (p_follower - 1e-4):
        prob_increase = min(prob_increase, 0.25)

    return round(float(final_price), 2), round(float(np.clip(prob_increase, 0.01, 0.99)), 4)


def generate_synthetic_competitor_dataset(
    processed_df: pd.DataFrame,
    n_samples: int = 4000,
    behavior: str = "adaptive",
    noise_std: float = 0.02,
    random_seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Generate labeled synthetic competitor response dataset from historical M5 context.

    Features generated:
    - leader_price (P_L)
    - leader_prev_price (P_L_prev)
    - leader_price_change_pct ((P_L - P_L_prev) / P_L_prev)
    - current_follower_price (P_F)
    - price_gap (P_L - P_F)
    - price_ratio (P_L / P_F)
    - recent_demand (lagged sales context from M5)
    - demand_trend (slope)
    - price_volatility (CV)
    - day_of_week, month, is_weekend, snap_CA, has_event

    Targets:
    - target_increase: Binary (1 if follower raises price, 0 if maintain/cut)
    - target_follower_price: Continuous dollar price
    - target_response_pct: Percentage change in follower price

    Args:
        processed_df: Processed M5 retail dataset for realistic baseline prices and context.
        n_samples: Number of synthetic scenarios to generate.
        behavior: Ground-truth rule ('adaptive', 'match', 'undercut', 'maintain', 'raise').
        noise_std: Noise standard deviation applied to synthetic response.
        random_seed: Reproducibility seed.

    Returns:
        pd.DataFrame containing synthetic features and ground-truth targets.
    """
    rng = np.random.RandomState(random_seed)

    # Sample realistic context rows from processed M5 data
    sample_rows = processed_df.sample(n=n_samples, replace=True, random_state=random_seed).copy()

    records = []
    for _, row in sample_rows.iterrows():
        base_price = float(row.get("sell_price", 2.0))
        # Leader tests price revision in [-25%, +25%] range around baseline
        pct_shock = rng.uniform(-0.25, 0.25)
        leader_prev_p = base_price
        leader_p = round(max(0.20, leader_prev_p * (1.0 + pct_shock)), 2)
        leader_pct_change = (leader_p - leader_prev_p) / leader_prev_p

        # Follower's existing price starts within [-15%, +15%] of leader baseline
        follower_p = round(max(0.20, leader_prev_p * rng.uniform(0.85, 1.15)), 2)

        # Generate ground-truth response under specified rule
        params = {"behavior": behavior, "noise_std": noise_std}
        follower_new_p, prob_increase = rule_based_response(
            leader_price=leader_p,
            current_follower_price=follower_p,
            params=params,
        )

        follower_resp_pct = (follower_new_p - follower_p) / follower_p
        target_increase = 1 if follower_new_p > follower_p else 0

        # Context features with clean fallback for early series NaNs
        recent_sales = float(row.get("rolling_mean_7", 10.0))
        if np.isnan(recent_sales):
            recent_sales = float(row.get("sales", 10.0)) if not np.isnan(float(row.get("sales", 10.0))) else 10.0

        sales_vol = float(row.get("rolling_std_7", 2.0))
        if np.isnan(sales_vol):
            sales_vol = 1.0

        lag1 = float(row.get("lag_sales_1", 10.0))
        lag7 = float(row.get("lag_sales_7", 10.0))
        trend = (lag1 - lag7) if (not np.isnan(lag1) and not np.isnan(lag7)) else 0.0

        records.append({
            "leader_price": leader_p,
            "leader_prev_price": leader_prev_p,
            "leader_price_change_pct": round(leader_pct_change, 4),
            "follower_price": follower_p,
            "price_gap": round(leader_p - follower_p, 4),
            "price_ratio": round(leader_p / follower_p, 4),
            "recent_demand": round(float(np.nan_to_num(recent_sales, nan=10.0)), 2),
            "sales_volatility": round(float(np.nan_to_num(sales_vol, nan=1.0)), 2),
            "demand_trend": round(float(np.nan_to_num(trend, nan=0.0)), 2),
            "day_of_week": int(row.get("day_of_week", 2)),
            "month": int(row.get("month", 5)),
            "is_weekend": int(row.get("is_weekend", 0)),
            "snap_CA": int(row.get("snap_CA", 0)),
            "has_event": int(row.get("has_event", 0)),
            "target_increase": int(target_increase),
            "target_follower_price": float(follower_new_p),
            "target_response_pct": round(float(follower_resp_pct), 4),
            "ground_truth_prob": round(float(prob_increase), 4),
            "simulation_label": SIMULATION_DISCLAIMER,
        })

    return pd.DataFrame(records)
