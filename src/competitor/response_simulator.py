"""Competitor response simulation module.

Simulates strategic competitor (follower) reaction logic under explicit behavioral
rules. Generates synthetic follower training datasets from known behavioral rules
with noise for subsequent ML modeling and rule-recovery benchmarking.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from src.utils.config import COMPETITOR_RESPONSE_NOISE, RANDOM_SEED

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
        # Adaptive response: blends price gap, leader change, and demand signal
        # If leader_prev_price provided in context, use true price revision
        if context and "leader_prev_price" in context:
            p_prev = max(0.01, float(context["leader_prev_price"]))
            leader_pct = (p_leader - p_prev) / p_prev
        else:
            leader_pct = (p_leader - p_follower) / p_follower

        gap_ratio = price_diff / p_follower
        recent_d = float(context.get("recent_demand", 20.0)) if context else 20.0
        z = 10.0 * leader_pct + 5.0 * gap_ratio + 0.01 * (recent_d - 20.0)
        prob_increase = float(1.0 / (1.0 + np.exp(-z)))

        # Magnitude reaction
        if leader_pct > 0:
            target_price = p_follower * (1.0 + resp_strength * leader_pct * 0.70)
        else:
            target_price = p_follower * (1.0 + resp_strength * leader_pct * 0.80)

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

    return round(float(final_price), 2), round(float(np.clip(prob_increase, 0.01, 0.99)), 4)


def generate_synthetic_competitor_dataset(
    processed_df: pd.DataFrame,
    n_samples: int = 5000,
    behavior: str = "adaptive",
    response_noise: float = COMPETITOR_RESPONSE_NOISE,
    noise_std: float = 0.05,
    random_seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Generate labeled synthetic competitor response dataset from historical M5 context.

    Features generated (given to the model):
    - leader_price (P_L)
    - leader_prev_price (P_L_prev)
    - leader_price_change_pct ((P_L - P_L_prev) / P_L_prev)
    - current_follower_price (P_F)
    - price_gap (P_L - P_F)
    - price_ratio (P_L / P_F)
    - recent_demand (lagged sales context from M5)
    - demand_trend (slope)
    - sales_volatility (rolling std)
    - day_of_week, month, is_weekend, snap_CA, has_event

    Unobserved latent factors (hidden from the model to induce realistic partial observability):
    - latent_inventory: Follower's private stock holding shock (~ N(0, 1))
    - latent_manager: Follower store manager's markup discretion shock (~ N(0, 0.5))

    Targets & Evaluation Ground Truth:
    - target_increase: Binary (1 if follower raises price, 0 if maintain/cut)
    - target_follower_price: Continuous dollar price sampled from response distribution
    - target_response_pct: Percentage change in follower price
    - ground_truth_prob: True Bernoulli parameter P(Increase)
    - bayes_optimal_pred: Theoretical Bayes-optimal classifier decision (P >= 0.50)

    Args:
        processed_df: Processed M5 retail dataset with 'date' for time-ordered sampling.
        n_samples: Number of synthetic scenarios to generate.
        behavior: Ground-truth rule ('adaptive', 'match', 'undercut', 'maintain', 'raise').
        response_noise: Noise std on response logit targeting ~78-83% Bayes ceiling.
        noise_std: Standard deviation for continuous magnitude noise.
        random_seed: Reproducibility seed.

    Returns:
        pd.DataFrame sorted chronologically by 'date'.
    """
    rng = np.random.RandomState(random_seed)

    # Time-ordered chronological selection from processed M5 data
    df_sorted = processed_df.sort_values(by=["date", "item_id", "store_id"]).reset_index(drop=True)
    indices = np.linspace(0, len(df_sorted) - 1, n_samples, dtype=int)
    sample_rows = df_sorted.iloc[indices].copy()

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

        price_gap = leader_p - follower_p
        gap_ratio = price_gap / follower_p

        # Hidden latent variables (unobserved by the model)
        latent_inv = rng.normal(0.0, 1.0)
        latent_mgr = rng.normal(0.0, 0.5)
        eps = rng.normal(0.0, response_noise)

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

        # Latent logit with features + unobserved latent factors + response noise
        z = (
            10.0 * leader_pct_change
            + 5.0 * gap_ratio
            + 0.01 * (recent_sales - 20.0)
            - 0.75 * latent_inv
            + latent_mgr
            + eps
        )
        prob_increase = float(1.0 / (1.0 + np.exp(-z)))
        prob_clipped = float(np.clip(prob_increase, 0.02, 0.98))

        # Bernoulli action draw (genuinely stochastic)
        target_increase = 1 if rng.uniform(0.0, 1.0) < prob_clipped else 0
        bayes_optimal_pred = 1 if prob_clipped >= 0.50 else 0

        # Continuous magnitude sampling with magnitude noise and latent factor
        mag_noise = rng.normal(0.0, noise_std)
        latent_mag = -0.025 * latent_inv + 0.015 * latent_mgr

        if target_increase == 1:
            resp_pct = max(
                0.005,
                0.65 * max(0.0, leader_pct_change)
                + 0.30 * max(0.0, gap_ratio)
                + latent_mag
                + mag_noise,
            )
        else:
            resp_pct = -max(
                0.0,
                -0.60 * min(0.0, leader_pct_change)
                - 0.30 * min(0.0, gap_ratio)
                - latent_mag
                - mag_noise,
            )

        follower_new_p = round(max(0.10, follower_p * (1.0 + resp_pct)), 2)
        follower_resp_pct = (follower_new_p - follower_p) / follower_p

        records.append({
            "date": pd.to_datetime(row["date"]),
            "leader_price": leader_p,
            "leader_prev_price": leader_prev_p,
            "leader_price_change_pct": round(leader_pct_change, 4),
            "follower_price": follower_p,
            "price_gap": round(price_gap, 4),
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
            "ground_truth_prob": round(float(prob_clipped), 4),
            "bayes_optimal_pred": int(bayes_optimal_pred),
            "simulation_label": SIMULATION_DISCLAIMER,
        })

    # Strict chronological order
    df_result = pd.DataFrame(records).sort_values("date").reset_index(drop=True)
    return df_result
