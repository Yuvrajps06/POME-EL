"""Strategy comparison and multi-period simulation evaluation framework.

Benchmarks Strategy A (Traditional Baseline), Strategy B (Game-Theoretic Rule Stackelberg),
and Strategy C (ML-Aware Stackelberg) against a ground-truth MarketEnvironment
using Common Random Numbers (CRN), reporting realized profit, downside risk,
confidence intervals, and honest win/loss/tie outcomes.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.competitor.response_model import CompetitorResponseModel, load_competitor_model
from src.experiments.simulation import MarketEnvironment
from src.optimization.baseline_optimizer import optimize_baseline
from src.optimization.game_optimizer import optimize_game_rule
from src.optimization.ml_optimizer import optimize_ml_stackelberg
from src.utils.config import DATA_PROCESSED_DIR, RANDOM_SEED


def run_strategy_trajectory(
    strategy: str,
    env: MarketEnvironment,
    n_periods: int = 28,
    a: float = 165.39,
    b: float = 79.18,
    c: float = 0.50,
    rule_params: Optional[Dict[str, Any]] = None,
    competitor_model: Optional[CompetitorResponseModel] = None,
    target_accuracy: Optional[float] = None,
    error_mode: str = "symmetric",
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Run a single multi-period simulation trajectory for one pricing strategy.

    Args:
        strategy: 'A' (Baseline), 'B' (Game Rule), or 'C' (ML-Aware).
        env: MarketEnvironment instance.
        n_periods: Simulation duration in days/periods.
        a: Assumed base demand intercept for optimizer.
        b: Assumed own-price sensitivity for optimizer.
        c: Assumed cross-price sensitivity for optimizer.
        rule_params: Follower parameters for Strategy B.
        competitor_model: Pre-loaded model for Strategy C.
        target_accuracy: Accuracy level for Strategy C.
        seed: Random seed for environment shocks.

    Returns:
        pd.DataFrame of period-by-period trajectory records.
    """
    env.reset(seed=seed)
    p_curr_leader = env.initial_leader_price
    p_curr_follower = env.initial_follower_price
    cost = env.cost

    records = []

    for t in range(1, n_periods + 1):
        # 1. Optimize price according to strategy
        if strategy.upper() == "A":
            strat_name = "Strategy A (Baseline)"
            opt = optimize_baseline(
                baseline_price=p_curr_leader,
                current_follower_price=p_curr_follower,
                cost=cost,
                a=a,
                b=b,
                c=c,
            )
        elif strategy.upper() == "B":
            strat_name = "Strategy B (Game Rule)"
            opt = optimize_game_rule(
                baseline_price=p_curr_leader,
                current_follower_price=p_curr_follower,
                cost=cost,
                a=a,
                b=b,
                c=c,
                rule_params=rule_params or {"behavior": "match"},
            )
        elif strategy.upper() == "C":
            strat_name = "Strategy C (ML-Aware)"
            opt = optimize_ml_stackelberg(
                baseline_price=p_curr_leader,
                current_follower_price=p_curr_follower,
                cost=cost,
                a=a,
                b=b,
                c=c,
                competitor_model=competitor_model,
                target_accuracy=target_accuracy,
                error_mode=error_mode,
            )
        else:
            raise ValueError(f"Unknown strategy code '{strategy}'. Must be 'A', 'B', or 'C'.")

        rec_price = opt.recommended_price

        # 2. Step the true-world market environment
        step_out = env.step(leader_price=rec_price)

        rec = {
            "period": t,
            "strategy": strat_name,
            "strategy_code": strategy.upper(),
            "recommended_price": rec_price,
            "follower_price_prev": step_out["follower_price_prev"],
            "realized_follower_price": step_out["realized_follower_price"],
            "realized_demand": step_out["realized_demand"],
            "realized_profit": step_out["realized_profit"],
            "realized_revenue": step_out["realized_revenue"],
            "unit_margin": step_out["unit_margin"],
            "cost": step_out["cost"],
            "sim_seed": seed,
        }
        records.append(rec)

        # Update firm state for next period
        p_curr_leader = rec_price
        p_curr_follower = step_out["realized_follower_price"]

    df_traj = pd.DataFrame(records)
    df_traj["cumulative_profit"] = df_traj["realized_profit"].cumsum()
    df_traj["cumulative_revenue"] = df_traj["realized_revenue"].cumsum()
    df_traj["cumulative_demand"] = df_traj["realized_demand"].cumsum()

    # Price change volatility: standard deviation of period-to-period price changes
    df_traj["price_diff"] = df_traj["recommended_price"].diff().fillna(0.0)

    return df_traj


def run_benchmark_experiments(
    initial_leader_price: float = 1.38,
    initial_follower_price: float = 1.37,
    cost: float = 0.83,
    a: float = 165.39,
    b: float = 79.18,
    c: float = 0.50,
    competitor_true_behavior: str = "adaptive",
    n_periods: int = 28,
    n_simulations: int = 40,
    rule_behavior: str = "match",
    target_accuracy: Optional[float] = None,
    master_seed: int = RANDOM_SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Execute rigorous multi-simulation benchmarking across Strategy A, B, and C.

    Uses Common Random Numbers (CRN) across strategies to isolate pricing policy differences
    from random environmental noise.

    Args:
        initial_leader_price: Baseline starting price.
        initial_follower_price: Baseline starting follower price.
        cost: Exogenous leader unit cost.
        a: Demand intercept.
        b: Own-price slope.
        c: Cross-price slope.
        competitor_true_behavior: True world competitor policy ('adaptive', 'undercut', 'match').
        n_periods: Days per simulation run.
        n_simulations: Repeated runs count.
        rule_behavior: Assumed behavior for Strategy B.
        target_accuracy: ML accuracy level for Strategy C.
        master_seed: Master seed.

    Returns:
        Tuple of (summary_df, all_trajectories_df, simulation_runs_df).
    """
    competitor_model = load_competitor_model()
    env = MarketEnvironment(
        initial_leader_price=initial_leader_price,
        initial_follower_price=initial_follower_price,
        cost=cost,
        a=a,
        b=b,
        c=c,
        competitor_behavior=competitor_true_behavior,
        random_seed=master_seed,
    )

    rng = np.random.RandomState(master_seed)
    seeds = [int(rng.randint(1, 1_000_000)) for _ in range(n_simulations)]

    all_traj_list = []
    run_records = []

    for sim_idx, s in enumerate(seeds):
        # Run all three strategies against the exact same seed (Common Random Numbers)
        traj_a = run_strategy_trajectory("A", env, n_periods, a, b, c, seed=s)
        traj_b = run_strategy_trajectory(
            "B", env, n_periods, a, b, c, rule_params={"behavior": rule_behavior}, seed=s
        )
        traj_c = run_strategy_trajectory(
            "C",
            env,
            n_periods,
            a,
            b,
            c,
            competitor_model=competitor_model,
            target_accuracy=target_accuracy,
            seed=s,
        )

        for t_df in [traj_a, traj_b, traj_c]:
            t_df["sim_run_id"] = sim_idx
            all_traj_list.append(t_df)

        prof_a = float(traj_a["realized_profit"].sum())
        prof_b = float(traj_b["realized_profit"].sum())
        prof_c = float(traj_c["realized_profit"].sum())

        run_records.append({
            "sim_run_id": sim_idx,
            "seed": s,
            "profit_A": prof_a,
            "profit_B": prof_b,
            "profit_C": prof_c,
            "improvement_B_over_A_pct": (prof_b - prof_a) / max(0.01, prof_a) * 100.0,
            "improvement_C_over_A_pct": (prof_c - prof_a) / max(0.01, prof_a) * 100.0,
            "winner": "C" if (prof_c > prof_a and prof_c >= prof_b) else ("B" if prof_b > prof_a else "A"),
        })

    all_trajectories_df = pd.concat(all_traj_list, ignore_index=True)
    runs_df = pd.DataFrame(run_records)

    # Summarize per strategy
    summaries = []
    strat_order = ["Strategy A (Baseline)", "Strategy B (Game Rule)", "Strategy C (ML-Aware)"]

    # Calculate baseline profit for percentage improvement
    base_mean_profit = runs_df["profit_A"].mean()

    for s_name in strat_order:
        s_traj = all_trajectories_df[all_trajectories_df["strategy"] == s_name]
        runs_prof = (
            runs_df["profit_A"]
            if "A" in s_name
            else (runs_df["profit_B"] if "B" in s_name else runs_df["profit_C"])
        )

        mean_tot_profit = float(runs_prof.mean())
        std_tot_profit = float(runs_prof.std())
        se_profit = std_tot_profit / np.sqrt(n_simulations)

        # 95% Confidence Interval for Total Profit
        ci_lower = mean_tot_profit - 1.96 * se_profit
        ci_upper = mean_tot_profit + 1.96 * se_profit

        # Profit improvement % vs Baseline A
        profit_imp_pct = (
            (mean_tot_profit - base_mean_profit) / base_mean_profit * 100.0
            if base_mean_profit > 0
            else 0.0
        )

        # Downside risk metric: 5th percentile of run profits
        p5_profit = float(np.percentile(runs_prof, 5))
        min_profit = float(runs_prof.min())

        # Price stability: std dev of price changes
        price_stability = float(s_traj["price_diff"].std())

        summaries.append({
            "strategy": s_name,
            "mean_total_profit": round(mean_tot_profit, 2),
            "ci_95_lower": round(ci_lower, 2),
            "ci_95_upper": round(ci_upper, 2),
            "profit_improvement_pct": round(profit_imp_pct, 3),
            "profit_std": round(std_tot_profit, 2),
            "downside_risk_p5": round(p5_profit, 2),
            "min_total_profit": round(min_profit, 2),
            "mean_period_profit": round(float(s_traj["realized_profit"].mean()), 2),
            "mean_period_revenue": round(float(s_traj["realized_revenue"].mean()), 2),
            "mean_period_demand": round(float(s_traj["realized_demand"].mean()), 2),
            "mean_price": round(float(s_traj["recommended_price"].mean()), 2),
            "price_volatility_std": round(price_stability, 4),
        })

    summary_df = pd.DataFrame(summaries)

    # Save to data/processed/experiments/
    exp_dir = DATA_PROCESSED_DIR / "experiments"
    exp_dir.mkdir(parents=True, exist_ok=True)

    summary_df.to_csv(exp_dir / "strategy_comparison_summary.csv", index=False)
    all_trajectories_df.to_csv(exp_dir / "strategy_trajectories.csv", index=False)
    runs_df.to_csv(exp_dir / "strategy_simulation_runs.csv", index=False)

    return summary_df, all_trajectories_df, runs_df


def plot_cumulative_profit(trajectories_df: pd.DataFrame) -> go.Figure:
    """Plot cumulative profit trajectories across simulation periods.

    Args:
        trajectories_df: Processed trajectory records from run_strategy_trajectory.

    Returns:
        Plotly Figure showing cumulative profit growth.
    """
    agg = (
        trajectories_df.groupby(["strategy", "period"], observed=True)["cumulative_profit"]
        .agg(mean="mean", std="std")
        .reset_index()
    )

    fig = px.line(
        agg,
        x="period",
        y="mean",
        color="strategy",
        title="Cumulative Realized Profit Across Multi-Period Horizon (True World Simulator)",
        labels={"period": "Simulation Day / Period", "mean": "Cumulative Profit ($)"},
        template="plotly_white",
        color_discrete_map={
            "Strategy A (Baseline)": "#1f77b4",
            "Strategy B (Game Rule)": "#ff7f0e",
            "Strategy C (ML-Aware)": "#2ca02c",
        },
    )
    return fig


def plot_strategy_comparison_bars(summary_df: pd.DataFrame) -> go.Figure:
    """Plot bar chart comparing total profits with 95% error bars across strategies.

    Args:
        summary_df: Summary DataFrame from run_benchmark_experiments.

    Returns:
        Plotly Figure.
    """
    error = summary_df["mean_total_profit"] - summary_df["ci_95_lower"]

    fig = px.bar(
        summary_df,
        x="strategy",
        y="mean_total_profit",
        error_y=error,
        color="strategy",
        title="Total Realized Profit Comparison with 95% Confidence Intervals",
        labels={"strategy": "Pricing Strategy", "mean_total_profit": "Mean Total Profit ($)"},
        template="plotly_white",
        text_auto=".1f",
        color_discrete_map={
            "Strategy A (Baseline)": "#1f77b4",
            "Strategy B (Game Rule)": "#ff7f0e",
            "Strategy C (ML-Aware)": "#2ca02c",
        },
    )
    return fig


def plot_price_trajectories(trajectories_df: pd.DataFrame) -> go.Figure:
    """Plot leader and realized follower prices over time across strategies.

    Args:
        trajectories_df: Trajectory DataFrame.

    Returns:
        Plotly Figure.
    """
    run0 = trajectories_df[trajectories_df["sim_run_id"] == 0].copy()
    fig = px.line(
        run0,
        x="period",
        y="recommended_price",
        color="strategy",
        title="Leader Price Trajectories over Time (Sample Simulation Run)",
        labels={"period": "Period", "recommended_price": "Leader Price ($)"},
        template="plotly_white",
    )
    return fig


def plot_profit_distribution_box(runs_df: pd.DataFrame) -> go.Figure:
    """Plot profit distribution box plots across repeated simulation runs.

    Args:
        runs_df: Runs DataFrame with profit_A, profit_B, profit_C.

    Returns:
        Plotly Figure.
    """
    melted = runs_df.melt(
        id_vars=["sim_run_id"],
        value_vars=["profit_A", "profit_B", "profit_C"],
        var_name="Strategy_Col",
        value_name="Total_Profit",
    )
    name_map = {
        "profit_A": "Strategy A (Baseline)",
        "profit_B": "Strategy B (Game Rule)",
        "profit_C": "Strategy C (ML-Aware)",
    }
    melted["Strategy"] = melted["Strategy_Col"].map(name_map)

    fig = px.box(
        melted,
        x="Strategy",
        y="Total_Profit",
        color="Strategy",
        points="all",
        title="Total Profit Distribution & Downside Variance Across Repeated Runs",
        labels={"Total_Profit": "Realized Profit ($)", "Strategy": "Strategy"},
        template="plotly_white",
        color_discrete_map={
            "Strategy A (Baseline)": "#1f77b4",
            "Strategy B (Game Rule)": "#ff7f0e",
            "Strategy C (ML-Aware)": "#2ca02c",
        },
    )
    return fig


def plot_revenue_comparison(trajectories_df: pd.DataFrame) -> go.Figure:
    """Plot cumulative revenue trajectories across simulation periods.

    Args:
        trajectories_df: Trajectory DataFrame.

    Returns:
        Plotly Figure.
    """
    agg = (
        trajectories_df.groupby(["strategy", "period"], observed=True)["cumulative_revenue"]
        .agg(mean="mean", std="std")
        .reset_index()
    )

    fig = px.line(
        agg,
        x="period",
        y="mean",
        color="strategy",
        title="Cumulative Realized Revenue Across Multi-Period Horizon",
        labels={"period": "Simulation Day / Period", "mean": "Cumulative Revenue ($)"},
        template="plotly_white",
        color_discrete_map={
            "Strategy A (Baseline)": "#1f77b4",
            "Strategy B (Game Rule)": "#ff7f0e",
            "Strategy C (ML-Aware)": "#2ca02c",
        },
    )
    return fig


def plot_sales_comparison(trajectories_df: pd.DataFrame) -> go.Figure:
    """Plot cumulative sales (units sold) across simulation periods.

    Args:
        trajectories_df: Trajectory DataFrame.

    Returns:
        Plotly Figure.
    """
    agg = (
        trajectories_df.groupby(["strategy", "period"], observed=True)["cumulative_demand"]
        .agg(mean="mean", std="std")
        .reset_index()
    )

    fig = px.line(
        agg,
        x="period",
        y="mean",
        color="strategy",
        title="Cumulative Units Sold Across Multi-Period Horizon",
        labels={"period": "Simulation Day / Period", "mean": "Cumulative Units Sold"},
        template="plotly_white",
        color_discrete_map={
            "Strategy A (Baseline)": "#1f77b4",
            "Strategy B (Game Rule)": "#ff7f0e",
            "Strategy C (ML-Aware)": "#2ca02c",
        },
    )
    return fig


def plot_improvement_distribution(runs_df: pd.DataFrame) -> go.Figure:
    """Plot profit improvement percentage distribution (Strategy B and C over A).

    Args:
        runs_df: Runs DataFrame containing improvement percentages.

    Returns:
        Plotly Figure.
    """
    melted = runs_df.melt(
        id_vars=["sim_run_id"],
        value_vars=["improvement_B_over_A_pct", "improvement_C_over_A_pct"],
        var_name="Comparison",
        value_name="Improvement_Pct",
    )
    name_map = {
        "improvement_B_over_A_pct": "Strategy B vs Baseline A",
        "improvement_C_over_A_pct": "Strategy C vs Baseline A",
    }
    melted["Comparison"] = melted["Comparison"].map(name_map)

    fig = px.box(
        melted,
        x="Comparison",
        y="Improvement_Pct",
        color="Comparison",
        points="all",
        title="Profit Improvement Percentage Distribution Over Strategy A",
        labels={"Improvement_Pct": "Profit Improvement (%)", "Comparison": "Strategy Comparison"},
        template="plotly_white",
        color_discrete_map={
            "Strategy B vs Baseline A": "#ff7f0e",
            "Strategy C vs Baseline A": "#2ca02c",
        },
    )
    fig.add_hline(y=0.0, line_dash="dash", line_color="gray", annotation_text="Break-even (0%)")
    return fig
