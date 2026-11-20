"""Stackelberg game theory engine.

Implements the analytical closed-form solution for the leader-follower pricing
game with linear demand and linear competitor reaction functions:
    Q_L = a - b * P_L + c * P_F + d * X
    P_F = alpha + beta * P_L
    P_L* = argmax (P_L - C_L) * Q_L(P_L, R(P_L))
"""

from typing import Any, Dict, Optional
import numpy as np


def closed_form_stackelberg(
    a: float,
    b: float,
    c: float,
    alpha: float,
    beta: float,
    cost: float,
    context_shift: float = 0.0,
) -> Dict[str, float]:
    """Calculate the closed-form analytical equilibrium for the Stackelberg game.

    Derivation:
        Follower Reaction: P_F = alpha + beta * P_L
        Leader Demand: Q_L = (a + context_shift + c * alpha) - (b - c * beta) * P_L
        Let A = a + context_shift + c * alpha
        Let B = b - c * beta
        Profit: pi_L = (P_L - C_L) * (A - B * P_L)
        d(pi_L)/d(P_L) = A - 2 * B * P_L + B * C_L = 0
        P_L* = (A + B * C_L) / (2 * B) = C_L / 2 + A / (2 * B)

    Args:
        a: Base demand intercept.
        b: Own-price sensitivity (b > 0).
        c: Cross-price sensitivity (c >= 0).
        alpha: Follower reaction intercept.
        beta: Follower reaction slope (d P_F / d P_L).
        cost: Leader unit cost (C_L).
        context_shift: Exogenous demand shifter (d * X).

    Returns:
        Dict with optimal_price, follower_price, demand, profit, and effective slopes.
    """
    net_intercept = float(a + context_shift + c * alpha)
    net_slope = float(b - c * beta)

    if net_slope <= 0.0:
        raise ValueError(
            f"Unstable reaction parameters: net slope (b - c * beta) = {net_slope:.4f} <= 0. "
            "Leader profit function is non-concave or unbounded."
        )

    # Analytical optimal price
    p_opt = (net_intercept + net_slope * cost) / (2.0 * net_slope)
    p_opt = max(0.0, float(p_opt))

    # Follower equilibrium response
    p_follower = float(alpha + beta * p_opt)

    # Equilibrium demand and profit
    q_opt = max(0.0, net_intercept - net_slope * p_opt)
    profit_opt = float((p_opt - cost) * q_opt) if q_opt > 0 else 0.0

    return {
        "optimal_price": round(p_opt, 4),
        "follower_price": round(p_follower, 4),
        "expected_demand": round(q_opt, 4),
        "expected_profit": round(profit_opt, 4),
        "effective_intercept": round(net_intercept, 4),
        "effective_slope": round(net_slope, 4),
        "cost": round(float(cost), 4),
    }


def closed_form_baseline_monopoly(
    a: float,
    b: float,
    cost: float,
    c: float = 0.0,
    constant_follower_price: float = 0.0,
    context_shift: float = 0.0,
) -> Dict[str, float]:
    """Calculate the closed-form benchmark monopoly price ignoring competitor feedback.

    Here beta = 0 (follower reaction slope is zero / competitor price is held fixed):
        P_L* = C_L / 2 + (a + context_shift + c * P_F) / (2 * b)

    Args:
        a: Base demand intercept.
        b: Own-price slope.
        cost: Leader unit cost.
        c: Cross-price slope (0 if competitor ignored).
        constant_follower_price: Assumed fixed follower price.
        context_shift: Exogenous demand shift.

    Returns:
        Dict with optimal monopoly price, demand, and profit.
    """
    return closed_form_stackelberg(
        a=a,
        b=b,
        c=c,
        alpha=constant_follower_price,
        beta=0.0,
        cost=cost,
        context_shift=context_shift,
    )
