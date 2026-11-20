"""Profit and margin calculation module.

Implements firm profit equations: pi = (P - C) * Q.
Strict adherence to Rule 2 of CONTEXT.md: Cost is an exogenous input parameter
and must NEVER be derived from selling price.
"""

from typing import Dict, Union


def calculate_profit(
    price: float,
    cost: float,
    demand: float,
) -> float:
    """Calculate total firm profit: pi = (P - C) * Q.

    Args:
        price: Selling price (P).
        cost: Unit cost parameter (C). Exogenous user/scenario input.
        demand: Expected quantity sold (Q >= 0).

    Returns:
        Firm profit. If demand is zero, profit is 0.0.
    """
    p = float(price)
    c = float(cost)
    q = max(0.0, float(demand))

    if q == 0.0:
        return 0.0

    unit_margin = p - c
    return float(unit_margin * q)


def calculate_margin(price: float, cost: float) -> float:
    """Calculate absolute dollar unit margin: P - C.

    Args:
        price: Selling price.
        cost: Exogenous unit cost.

    Returns:
        Dollar margin per unit. Negative if price < cost.
    """
    return float(price - cost)


def calculate_margin_pct(price: float, cost: float) -> float:
    """Calculate percentage gross margin: (P - C) / P.

    Args:
        price: Selling price.
        cost: Exogenous unit cost.

    Returns:
        Margin percentage as float. Returns 0.0 if price is zero.
    """
    p = float(price)
    if p <= 0.0:
        return 0.0
    return float((p - cost) / p)


def summarize_financials(
    price: float,
    cost: float,
    demand: float,
) -> Dict[str, float]:
    """Summarize revenue, profit, unit margin, and margin percentage.

    Args:
        price: Selling price.
        cost: Exogenous unit cost.
        demand: Units sold.

    Returns:
        Dict of financial metrics.
    """
    q = max(0.0, float(demand))
    rev = float(price * q)
    profit = calculate_profit(price, cost, q)
    margin = calculate_margin(price, cost)
    margin_pct = calculate_margin_pct(price, cost)

    return {
        "price": round(float(price), 2),
        "cost": round(float(cost), 2),
        "demand": round(float(q), 2),
        "revenue": round(rev, 2),
        "profit": round(profit, 2),
        "unit_margin": round(margin, 2),
        "margin_pct": round(margin_pct, 4),
        "has_negative_margin": bool(margin < 0.0),
    }
