"""Game engine demand functions.

Implements structural linear demand specifications with own-price and cross-price
elasticity components: Q_L = a - b * P_L + c * P_F + d * X, clipped at Q >= 0.
"""

from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd


def linear_demand(
    leader_price: float,
    follower_price: float,
    a: float,
    b: float,
    c: float = 0.50,
    context_shift: float = 0.0,
) -> float:
    """Compute expected leader demand under structural linear specification.

    Formula:
        Q_L = max(0.0, a - b * P_L + c * P_F + context_shift)

    Args:
        leader_price: Leader selling price (P_L).
        follower_price: Competitor follower selling price (P_F).
        a: Base demand intercept (from Phase 3 linear approximation).
        b: Own-price sensitivity slope (b > 0).
        c: Cross-price sensitivity slope (c >= 0, substitute goods).
        context_shift: Exogenous demand shift from calendar, holidays, or SNAP (d * X).

    Returns:
        Non-negative expected demand quantity Q_L.
    """
    p_l = max(0.0, float(leader_price))
    p_f = max(0.0, float(follower_price))

    raw_q = float(a) - float(b) * p_l + float(c) * p_f + float(context_shift)
    return float(max(0.0, raw_q))


def calculate_context_shift(
    context: Optional[Dict[str, Any]] = None,
    coefficients: Optional[Dict[str, float]] = None,
) -> float:
    """Calculate exogenous demand shift from context features (d * X).

    Args:
        context: Context feature dictionary (day_of_week, snap_CA, has_event).
        coefficients: Regression weights for each feature.

    Returns:
        Scalar exogenous demand shift d * X.
    """
    if not context or not coefficients:
        return 0.0

    shift = 0.0
    for feat, weight in coefficients.items():
        val = float(context.get(feat, 0.0))
        shift += weight * val
    return shift
