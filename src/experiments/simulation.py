"""Ground-truth market environment simulator for strategy benchmarking.

Simulates a 'true world' retail market where customer demand and competitor price
adjustments realize stochastically according to true parameters.
Evaluates realized profit and revenue for any pricing strategy (Baseline, Game, ML-aware).
"""

from typing import Any, Dict, Optional, Tuple, Union
import numpy as np

from src.competitor.response_simulator import (
    SIMULATION_DISCLAIMER,
    rule_based_response,
)
from src.game.demand_function import linear_demand
from src.game.profit_function import calculate_margin, calculate_profit
from src.utils.config import RANDOM_SEED


class MarketEnvironment:
    """True-world market simulation environment."""

    def __init__(
        self,
        initial_leader_price: float,
        initial_follower_price: float,
        cost: float,
        a: float,
        b: float,
        c: float = 0.50,
        competitor_behavior: str = "adaptive",
        competitor_params: Optional[Dict[str, Any]] = None,
        demand_noise_std: float = 2.0,
        competitor_noise_std: float = 0.02,
        random_seed: int = RANDOM_SEED,
    ):
        """Initialize market environment.

        Args:
            initial_leader_price: Starting leader price.
            initial_follower_price: Starting competitor follower price.
            cost: Exogenous leader unit cost.
            a: True ground-truth demand intercept.
            b: True ground-truth own-price sensitivity slope.
            c: True ground-truth cross-price sensitivity slope.
            competitor_behavior: True competitor behavioral policy ('adaptive', 'match', 'undercut', etc.).
            competitor_params: Optional overrides for competitor reaction policy.
            demand_noise_std: Standard deviation of unobserved demand shocks.
            competitor_noise_std: Standard deviation of competitor pricing execution noise.
            random_seed: Master random seed for reproducibility.
        """
        self.initial_leader_price = float(initial_leader_price)
        self.initial_follower_price = float(initial_follower_price)
        self.cost = float(cost)
        self.a = float(a)
        self.b = float(b)
        self.c = float(c)
        self.competitor_behavior = str(competitor_behavior).lower()
        self.competitor_params = competitor_params or {}
        self.demand_noise_std = float(demand_noise_std)
        self.competitor_noise_std = float(competitor_noise_std)
        self.random_seed = int(random_seed)

        self.rng = np.random.RandomState(self.random_seed)
        self.current_follower_price = self.initial_follower_price
        self.period = 0

    def reset(self, seed: Optional[int] = None) -> float:
        """Reset environment to initial conditions.

        Args:
            seed: Optional seed override.

        Returns:
            Initial follower price.
        """
        if seed is not None:
            self.random_seed = seed
        self.rng = np.random.RandomState(self.random_seed)
        self.current_follower_price = self.initial_follower_price
        self.period = 0
        return self.current_follower_price

    def step(
        self,
        leader_price: float,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Execute one simulation period given the leader's recommended price.

        Steps:
        1. True competitor reacts based on true behavioral policy + execution noise.
        2. True customer demand realizes based on true parameters + demand shock.
        3. Realized financial outcomes (profit, revenue, margin) are recorded.
        4. Environment state advances.

        Args:
            leader_price: Strategy recommended price.
            context: Environment covariates (seasonality, SNAP, events).

        Returns:
            Dict containing realized prices, demand, profit, revenue, and margins.
        """
        self.period += 1
        p_lead = max(0.10, float(leader_price))
        p_foll_prev = self.current_follower_price

        # 1. True competitor reaction
        params = dict(self.competitor_params)
        params["behavior"] = self.competitor_behavior
        params["noise_std"] = 0.0  # Rule returns base reaction; apply noise via self.rng for exact reproducibility

        # Draw competitor reaction and apply stochastic execution noise with self.rng
        raw_foll_p, prob_inc = rule_based_response(
            leader_price=p_lead,
            current_follower_price=p_foll_prev,
            context=context,
            params=params,
        )

        comp_noise = self.rng.normal(0.0, self.competitor_noise_std * raw_foll_p)
        realized_follower_price = max(0.10, round(raw_foll_p + comp_noise, 2))

        # 2. True demand realization
        ctx_shift = float(context.get("context_shift", 0.0)) if context else 0.0
        demand_shock = self.rng.normal(0.0, self.demand_noise_std)

        expected_q = linear_demand(
            leader_price=p_lead,
            follower_price=realized_follower_price,
            a=self.a,
            b=self.b,
            c=self.c,
            context_shift=ctx_shift,
        )
        realized_demand = max(0.0, round(float(expected_q + demand_shock), 2))

        # 3. Realized financials
        realized_profit = calculate_profit(p_lead, self.cost, realized_demand)
        realized_revenue = float(p_lead * realized_demand)
        unit_margin = calculate_margin(p_lead, self.cost)

        # 4. Advance state
        self.current_follower_price = realized_follower_price

        return {
            "period": self.period,
            "leader_price": round(p_lead, 2),
            "follower_price_prev": round(p_foll_prev, 2),
            "realized_follower_price": round(realized_follower_price, 2),
            "expected_demand": round(float(expected_q), 2),
            "realized_demand": round(realized_demand, 2),
            "realized_profit": round(float(realized_profit), 2),
            "realized_revenue": round(float(realized_revenue), 2),
            "unit_margin": round(float(unit_margin), 2),
            "cost": round(self.cost, 2),
            "demand_shock": round(float(demand_shock), 3),
        }
