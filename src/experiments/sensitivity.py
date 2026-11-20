"""Phase 7: Sensitivity Analysis & Simulation Experiments.

Systematic parameter sweeps across:
1. Competitor response strength (weak/moderate/strong) x ML accuracy (50%-90%).
2. ML accuracy curves under symmetric vs asymmetric errors (optimistic vs pessimistic)
   with break-even threshold estimation where Strategy C beats Baseline A.
3. Competitor behavior (match, undercut, maintain, raise) and delay (0, 1, 3 periods).
4. Demand sensitivity: own-price slope (low/med/high) and cross-price slope.
5. Robustness under parameter uncertainty (cost and demand misspecification).

All sweeps support caching to disk for rapid dashboard rendering.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.competitor.response_model import CompetitorResponseModel, load_competitor_model
from src.experiments.comparison import run_strategy_trajectory
from src.experiments.simulation import MarketEnvironment
from src.optimization.baseline_optimizer import optimize_baseline
from src.optimization.game_optimizer import optimize_game_rule
from src.optimization.ml_optimizer import optimize_ml_stackelberg
from src.utils.config import DATA_PROCESSED_DIR, RANDOM_SEED

EXPERIMENTS_DIR = DATA_PROCESSED_DIR / "experiments"


def sweep_accuracy_and_response_strength(
    n_simulations: int = 15,
    n_periods: int = 14,
    competitor_model: Optional[CompetitorResponseModel] = None,
    master_seed: int = RANDOM_SEED,
    cache_path: Optional[Path] = None,
    force_recompute: bool = False,
) -> pd.DataFrame:
    """Sweep competitor response strength x ML accuracy to evaluate profit improvement % heatmap.

    Args:
        n_simulations: Number of repeated seeds per cell.
        n_periods: Simulation days per trajectory.
        competitor_model: Pre-loaded CompetitorResponseModel.
        master_seed: Seed for CRN.
        cache_path: Path to cache CSV.
        force_recompute: If True, re-runs sweep even if cache exists.

    Returns:
        pd.DataFrame with response_strength, target_accuracy, mean_profit_A, mean_profit_C, improvement_pct.
    """
    save_file = cache_path or (EXPERIMENTS_DIR / "sensitivity_heatmap_strength_accuracy.csv")
    if save_file.exists() and not force_recompute:
        return pd.read_csv(save_file)

    model = competitor_model or load_competitor_model()
    strengths = [0.25, 0.75, 1.50]  # weak, moderate, strong
    strength_labels = {0.25: "Weak (0.25)", 0.75: "Moderate (0.75)", 1.50: "Strong (1.50)"}
    accuracies = [0.50, 0.60, 0.70, 0.80, 0.90]

    rng = np.random.RandomState(master_seed)
    seeds = [int(rng.randint(1, 1_000_000)) for _ in range(n_simulations)]

    records = []

    for resp_str in strengths:
        for acc in accuracies:
            prof_a_list = []
            prof_c_list = []

            for s in seeds:
                env = MarketEnvironment(
                    initial_leader_price=1.38,
                    initial_follower_price=1.37,
                    cost=0.83,
                    a=165.39,
                    b=79.18,
                    c=0.50,
                    competitor_behavior="adaptive",
                    competitor_params={"response_strength": resp_str},
                    random_seed=s,
                )

                traj_a = run_strategy_trajectory("A", env, n_periods=n_periods, seed=s)
                traj_c = run_strategy_trajectory(
                    "C",
                    env,
                    n_periods=n_periods,
                    competitor_model=model,
                    target_accuracy=acc,
                    seed=s,
                )

                prof_a_list.append(float(traj_a["realized_profit"].sum()))
                prof_c_list.append(float(traj_c["realized_profit"].sum()))

            mean_a = float(np.mean(prof_a_list))
            mean_c = float(np.mean(prof_c_list))
            imp_pct = (mean_c - mean_a) / max(0.01, mean_a) * 100.0

            records.append({
                "response_strength": resp_str,
                "strength_label": strength_labels[resp_str],
                "target_accuracy": acc,
                "accuracy_pct": f"{int(acc * 100)}%",
                "mean_profit_A": round(mean_a, 2),
                "mean_profit_C": round(mean_c, 2),
                "profit_difference": round(mean_c - mean_a, 2),
                "profit_improvement_pct": round(imp_pct, 4),
            })

    df = pd.DataFrame(records)
    save_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_file, index=False)
    return df


def sweep_accuracy_curves(
    n_simulations: int = 20,
    n_periods: int = 14,
    competitor_model: Optional[CompetitorResponseModel] = None,
    master_seed: int = RANDOM_SEED,
    cache_path: Optional[Path] = None,
    force_recompute: bool = False,
) -> pd.DataFrame:
    """Sweep ML accuracy vs profit curve under symmetric, optimistic, and pessimistic error regimes.

    Estimates break-even accuracy where Strategy C overtakes Baseline A.

    Args:
        n_simulations: Number of repeated seeds per evaluation.
        n_periods: Days per simulation.
        competitor_model: Pre-loaded CompetitorResponseModel.
        master_seed: Master seed for CRN.
        cache_path: File path for caching.
        force_recompute: Overwrite existing cache if True.

    Returns:
        pd.DataFrame with error_mode, target_accuracy, mean_profit_A, mean_profit_C, improvement_pct.
    """
    save_file = cache_path or (EXPERIMENTS_DIR / "sensitivity_accuracy_curve.csv")
    if save_file.exists() and not force_recompute:
        return pd.read_csv(save_file)

    model = competitor_model or load_competitor_model()
    accuracies = [0.50, 0.60, 0.70, 0.80, 0.90]
    error_modes = ["symmetric", "optimistic", "pessimistic"]

    rng = np.random.RandomState(master_seed)
    seeds = [int(rng.randint(1, 1_000_000)) for _ in range(n_simulations)]

    records = []

    for mode in error_modes:
        for acc in accuracies:
            prof_a_list = []
            prof_c_list = []

            for s in seeds:
                env = MarketEnvironment(
                    initial_leader_price=1.38,
                    initial_follower_price=1.37,
                    cost=0.83,
                    a=165.39,
                    b=79.18,
                    c=0.50,
                    competitor_behavior="adaptive",
                    random_seed=s,
                )

                traj_a = run_strategy_trajectory("A", env, n_periods=n_periods, seed=s)
                traj_c = run_strategy_trajectory(
                    "C",
                    env,
                    n_periods=n_periods,
                    competitor_model=model,
                    target_accuracy=acc,
                    error_mode=mode,
                    seed=s,
                )

                prof_a_list.append(float(traj_a["realized_profit"].sum()))
                prof_c_list.append(float(traj_c["realized_profit"].sum()))

            mean_a = float(np.mean(prof_a_list))
            mean_c = float(np.mean(prof_c_list))
            imp_pct = (mean_c - mean_a) / max(0.01, mean_a) * 100.0

            records.append({
                "error_mode": mode,
                "target_accuracy": acc,
                "accuracy_pct": int(acc * 100),
                "mean_profit_A": round(mean_a, 2),
                "mean_profit_C": round(mean_c, 2),
                "profit_improvement_pct": round(imp_pct, 4),
                "c_wins_pct": round(float(np.mean(np.array(prof_c_list) >= np.array(prof_a_list)) * 100.0), 1),
            })

    df = pd.DataFrame(records)
    save_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_file, index=False)
    return df


def sweep_competitor_behaviors(
    n_simulations: int = 15,
    n_periods: int = 14,
    competitor_model: Optional[CompetitorResponseModel] = None,
    master_seed: int = RANDOM_SEED,
    cache_path: Optional[Path] = None,
    force_recompute: bool = False,
) -> pd.DataFrame:
    """Sweep competitor behavioral policies and reaction delays (0, 1, 3 periods).

    Evaluates realized profit under match, undercut, maintain, and raise behaviors.

    Args:
        n_simulations: Repeated runs.
        n_periods: Horizon length.
        competitor_model: Pre-loaded CompetitorResponseModel.
        master_seed: CRN seed.
        cache_path: Cache file path.
        force_recompute: Recalculate flag.

    Returns:
        pd.DataFrame with behavior, delay, profit_A, profit_B, profit_C, imp_C_vs_A.
    """
    save_file = cache_path or (EXPERIMENTS_DIR / "sensitivity_competitor_behaviors.csv")
    if save_file.exists() and not force_recompute:
        return pd.read_csv(save_file)

    model = competitor_model or load_competitor_model()
    behaviors = ["match", "undercut", "maintain", "raise"]
    delays = [0, 1, 3]

    rng = np.random.RandomState(master_seed)
    seeds = [int(rng.randint(1, 1_000_000)) for _ in range(n_simulations)]

    records = []

    for beh in behaviors:
        for delay in delays:
            prof_a, prof_b, prof_c = [], [], []

            for s in seeds:
                env = MarketEnvironment(
                    initial_leader_price=1.38,
                    initial_follower_price=1.37,
                    cost=0.83,
                    a=165.39,
                    b=79.18,
                    c=0.50,
                    competitor_behavior=beh,
                    competitor_delay=delay,
                    random_seed=s,
                )

                traj_a = run_strategy_trajectory("A", env, n_periods=n_periods, seed=s)
                traj_b = run_strategy_trajectory(
                    "B", env, n_periods=n_periods, rule_params={"behavior": beh}, seed=s
                )
                traj_c = run_strategy_trajectory(
                    "C", env, n_periods=n_periods, competitor_model=model, seed=s
                )

                prof_a.append(float(traj_a["realized_profit"].sum()))
                prof_b.append(float(traj_b["realized_profit"].sum()))
                prof_c.append(float(traj_c["realized_profit"].sum()))

            mean_a = float(np.mean(prof_a))
            mean_b = float(np.mean(prof_b))
            mean_c = float(np.mean(prof_c))

            records.append({
                "competitor_behavior": beh,
                "delay_periods": delay,
                "mean_profit_A": round(mean_a, 2),
                "mean_profit_B": round(mean_b, 2),
                "mean_profit_C": round(mean_c, 2),
                "improvement_B_over_A_pct": round((mean_b - mean_a) / max(0.01, mean_a) * 100.0, 3),
                "improvement_C_over_A_pct": round((mean_c - mean_a) / max(0.01, mean_a) * 100.0, 3),
            })

    df = pd.DataFrame(records)
    save_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_file, index=False)
    return df


def sweep_demand_sensitivity(
    cache_path: Optional[Path] = None,
    force_recompute: bool = False,
) -> pd.DataFrame:
    """Sweep demand sensitivity: own-price slope (b) and cross-price slope (c) vs optimal prices.

    Evaluates optimal pricing decisions across own-price sensitivity (low, med, high)
    and cross-price elasticity (c in [0.0, 1.0]).

    Args:
        cache_path: Cache file path.
        force_recompute: Recalculate flag.

    Returns:
        pd.DataFrame with b_level, b, c, optimal_price_A, optimal_price_B, optimal_price_C.
    """
    save_file = cache_path or (EXPERIMENTS_DIR / "sensitivity_demand_slopes.csv")
    if save_file.exists() and not force_recompute:
        return pd.read_csv(save_file)

    b_cases = [
        ("Low Elasticity", 40.0),
        ("Medium Elasticity (Empirical)", 79.18),
        ("High Elasticity", 120.0),
    ]
    c_values = [0.00, 0.25, 0.50, 0.75, 1.00]
    cost = 0.83
    base_p = 1.38
    p_foll = 1.37
    a = 165.39

    model = load_competitor_model()
    records = []

    for b_label, b_val in b_cases:
        for c_val in c_values:
            opt_a = optimize_baseline(base_p, p_foll, cost, a, b_val, c=c_val)
            opt_b = optimize_game_rule(base_p, p_foll, cost, a, b_val, c=c_val, rule_params={"behavior": "match"})
            opt_c = optimize_ml_stackelberg(base_p, p_foll, cost, a, b_val, c=c_val, competitor_model=model)

            records.append({
                "b_level": b_label,
                "own_price_b": b_val,
                "cross_price_c": c_val,
                "price_A": opt_a.recommended_price,
                "price_B": opt_b.recommended_price,
                "price_C": opt_c.recommended_price,
                "profit_A": opt_a.expected_profit,
                "profit_B": opt_b.expected_profit,
                "profit_C": opt_c.expected_profit,
                "price_spread_C_minus_A": round(opt_c.recommended_price - opt_a.recommended_price, 3),
            })

    df = pd.DataFrame(records)
    save_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_file, index=False)
    return df


def sweep_robustness_uncertainty(
    n_simulations: int = 15,
    n_periods: int = 14,
    competitor_model: Optional[CompetitorResponseModel] = None,
    master_seed: int = RANDOM_SEED,
    cache_path: Optional[Path] = None,
    force_recompute: bool = False,
) -> pd.DataFrame:
    """Sweep performance under parameter uncertainty: cost and demand volatility.

    Tests decision robustness when the firm's assumed unit cost or demand parameters
    deviate from the true-world values.

    Args:
        n_simulations: Repeated runs.
        n_periods: Days per simulation.
        competitor_model: Pre-loaded CompetitorResponseModel.
        master_seed: Master seed.
        cache_path: Cache file path.
        force_recompute: Recalculate flag.

    Returns:
        pd.DataFrame with cost_scenario, demand_volatility, mean_profit_A, mean_profit_C.
    """
    save_file = cache_path or (EXPERIMENTS_DIR / "sensitivity_robustness.csv")
    if save_file.exists() and not force_recompute:
        return pd.read_csv(save_file)

    model = competitor_model or load_competitor_model()
    costs = [0.65, 0.83, 1.05]
    volatilities = [0.5, 2.0, 5.0]  # low, empirical, high unobserved demand noise

    rng = np.random.RandomState(master_seed)
    seeds = [int(rng.randint(1, 1_000_000)) for _ in range(n_simulations)]

    records = []

    for c_unit in costs:
        for vol in volatilities:
            prof_a, prof_c = [], []

            for s in seeds:
                env = MarketEnvironment(
                    initial_leader_price=1.38,
                    initial_follower_price=1.37,
                    cost=c_unit,
                    a=165.39,
                    b=79.18,
                    c=0.50,
                    competitor_behavior="adaptive",
                    demand_noise_std=vol,
                    random_seed=s,
                )

                traj_a = run_strategy_trajectory("A", env, n_periods=n_periods, seed=s)
                traj_c = run_strategy_trajectory(
                    "C", env, n_periods=n_periods, competitor_model=model, seed=s
                )

                prof_a.append(float(traj_a["realized_profit"].sum()))
                prof_c.append(float(traj_c["realized_profit"].sum()))

            mean_a = float(np.mean(prof_a))
            mean_c = float(np.mean(prof_c))

            vol_label = "Low (0.5)" if vol < 1.0 else ("Moderate (2.0)" if vol < 3.0 else "High (5.0)")

            records.append({
                "unit_cost": c_unit,
                "demand_volatility": vol,
                "volatility_label": vol_label,
                "mean_profit_A": round(mean_a, 2),
                "mean_profit_C": round(mean_c, 2),
                "profit_difference": round(mean_c - mean_a, 2),
                "profit_improvement_pct": round((mean_c - mean_a) / max(0.01, mean_a) * 100.0, 4),
            })

    df = pd.DataFrame(records)
    save_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_file, index=False)
    return df


def run_all_sensitivity_sweeps(
    n_simulations: int = 15,
    n_periods: int = 14,
    force_recompute: bool = False,
) -> Dict[str, pd.DataFrame]:
    """Execute all Phase 7 sensitivity analysis sweeps and cache outputs to disk.

    Args:
        n_simulations: Simulation count per cell.
        n_periods: Multi-period simulation horizon.
        force_recompute: Recalculate flag.

    Returns:
        Dict mapping sweep names to result DataFrames.
    """
    model = load_competitor_model()

    df_heatmap = sweep_accuracy_and_response_strength(
        n_simulations=n_simulations,
        n_periods=n_periods,
        competitor_model=model,
        force_recompute=force_recompute,
    )
    df_accuracy = sweep_accuracy_curves(
        n_simulations=n_simulations,
        n_periods=n_periods,
        competitor_model=model,
        force_recompute=force_recompute,
    )
    df_behaviors = sweep_competitor_behaviors(
        n_simulations=n_simulations,
        n_periods=n_periods,
        competitor_model=model,
        force_recompute=force_recompute,
    )
    df_demand = sweep_demand_sensitivity(
        force_recompute=force_recompute,
    )
    df_robustness = sweep_robustness_uncertainty(
        n_simulations=n_simulations,
        n_periods=n_periods,
        competitor_model=model,
        force_recompute=force_recompute,
    )

    return {
        "heatmap": df_heatmap,
        "accuracy_curve": df_accuracy,
        "behaviors": df_behaviors,
        "demand": df_demand,
        "robustness": df_robustness,
    }


# =========================================================================
# Plotly Visualization Functions
# =========================================================================

def plot_heatmap_strength_x_accuracy(df_heatmap: pd.DataFrame) -> go.Figure:
    """Generate 2D heatmap of profit improvement % over response strength x ML accuracy.

    Args:
        df_heatmap: DataFrame from sweep_accuracy_and_response_strength.

    Returns:
        Plotly Figure.
    """
    pivot = df_heatmap.pivot(
        index="strength_label",
        columns="accuracy_pct",
        values="profit_improvement_pct",
    )
    # Ensure correct ordering
    acc_cols = [c for c in ["50%", "60%", "70%", "80%", "90%"] if c in pivot.columns]
    str_idx = [i for i in ["Weak (0.25)", "Moderate (0.75)", "Strong (1.50)"] if i in pivot.index]
    pivot = pivot.loc[str_idx, acc_cols]

    fig = px.imshow(
        pivot,
        labels=dict(x="ML Competitor Model Accuracy", y="Competitor Response Strength", color="Profit Imp. (%)"),
        x=pivot.columns,
        y=pivot.index,
        color_continuous_scale="RdYlGn",
        title="Profit Improvement % (Strategy C vs A) by Competitor Response Strength & ML Accuracy",
        text_auto=".3f",
        template="plotly_white",
    )
    fig.update_layout(coloraxis_colorbar=dict(title="Gain %"))
    return fig


def plot_accuracy_vs_profit_curve(df_accuracy: pd.DataFrame) -> go.Figure:
    """Plot ML accuracy vs profit improvement curve with break-even threshold and asymmetric error comparison.

    Args:
        df_accuracy: DataFrame from sweep_accuracy_curves.

    Returns:
        Plotly Figure.
    """
    fig = px.line(
        df_accuracy,
        x="accuracy_pct",
        y="profit_improvement_pct",
        color="error_mode",
        markers=True,
        title="Impact of ML Accuracy on Realized Profit Improvement: Symmetric vs Asymmetric Errors",
        labels={
            "accuracy_pct": "Competitor Model Accuracy (%)",
            "profit_improvement_pct": "Profit Improvement vs Strategy A (%)",
            "error_mode": "Error Regime",
        },
        template="plotly_white",
        color_discrete_map={
            "symmetric": "#1f77b4",
            "optimistic": "#d62728",
            "pessimistic": "#2ca02c",
        },
    )

    # Add break-even threshold horizontal line
    fig.add_hline(
        y=0.0,
        line_dash="dash",
        line_color="black",
        annotation_text="Break-even with Baseline A (0.0%)",
        annotation_position="bottom right",
    )

    fig.update_layout(
        xaxis=dict(tickmode="array", tickvals=[50, 60, 70, 80, 90]),
        yaxis=dict(ticksuffix="%"),
    )
    return fig


def plot_competitor_behavior_comparison(df_behaviors: pd.DataFrame) -> go.Figure:
    """Plot realized profits across competitor behavioral policies (match, undercut, maintain, raise).

    Args:
        df_behaviors: DataFrame from sweep_competitor_behaviors.

    Returns:
        Plotly Figure.
    """
    # Group across delay = 0 for clean bar representation
    sub = df_behaviors[df_behaviors["delay_periods"] == 0].copy()
    melted = sub.melt(
        id_vars=["competitor_behavior"],
        value_vars=["mean_profit_A", "mean_profit_B", "mean_profit_C"],
        var_name="Strategy_Col",
        value_name="Realized_Profit",
    )
    name_map = {
        "mean_profit_A": "Strategy A (Baseline)",
        "mean_profit_B": "Strategy B (Game Rule)",
        "mean_profit_C": "Strategy C (ML-Aware)",
    }
    melted["Strategy"] = melted["Strategy_Col"].map(name_map)

    fig = px.bar(
        melted,
        x="competitor_behavior",
        y="Realized_Profit",
        color="Strategy",
        barmode="group",
        title="Realized Profit by Competitor Behavioral Policy (Delay = 0)",
        labels={
            "competitor_behavior": "True Competitor Behavioral Policy",
            "Realized_Profit": "Realized Profit ($)",
            "Strategy": "Pricing Strategy",
        },
        template="plotly_white",
        color_discrete_map={
            "Strategy A (Baseline)": "#1f77b4",
            "Strategy B (Game Rule)": "#ff7f0e",
            "Strategy C (ML-Aware)": "#2ca02c",
        },
    )
    return fig


def plot_demand_sensitivity_vs_price(df_demand: pd.DataFrame) -> go.Figure:
    """Plot demand elasticity parameters vs optimal recommended prices.

    Args:
        df_demand: DataFrame from sweep_demand_sensitivity.

    Returns:
        Plotly Figure.
    """
    fig = px.line(
        df_demand,
        x="cross_price_c",
        y="price_C",
        color="b_level",
        markers=True,
        title="Recommended Price (Strategy C) across Cross-Price Sensitivity (c) & Own Elasticity (b)",
        labels={
            "cross_price_c": "Cross-Price Sensitivity Slope (c)",
            "price_C": "Recommended Price P_L* ($)",
            "b_level": "Own-Price Sensitivity Level (b)",
        },
        template="plotly_white",
    )
    fig.update_layout(yaxis=dict(tickprefix="$"))
    return fig


def plot_robustness_analysis(df_robustness: pd.DataFrame) -> go.Figure:
    """Plot robustness of Strategy C vs A across cost scenarios and demand volatility.

    Args:
        df_robustness: DataFrame from sweep_robustness_uncertainty.

    Returns:
        Plotly Figure.
    """
    fig = px.bar(
        df_robustness,
        x="volatility_label",
        y="profit_difference",
        color="unit_cost",
        barmode="group",
        title="Strategy C Profit Advantage ($) Across Cost Levels & Demand Volatility",
        labels={
            "volatility_label": "Unobserved Demand Volatility Level",
            "profit_difference": "Profit Advantage (C - A) ($)",
            "unit_cost": "Leader Unit Cost ($)",
        },
        template="plotly_white",
    )
    fig.add_hline(y=0.0, line_dash="dash", line_color="black")
    return fig
