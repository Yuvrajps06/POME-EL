"""Script to run Phase 6 Strategy Benchmarking Experiment and save results."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.experiments.comparison import (
    run_benchmark_experiments,
    plot_cumulative_profit,
    plot_strategy_comparison_bars,
    plot_price_trajectories,
    plot_profit_distribution_box,
    plot_revenue_comparison,
    plot_sales_comparison,
    plot_improvement_distribution,
)


def main():
    print("=" * 60)
    print("Running Phase 6: Strategy Benchmarking Simulation Experiment")
    print("=" * 60)

    # FOODS_3_090: Leader CA_1, Cost = $0.83, P_L0 = $1.38, P_F0 = $1.37
    # a = 165.39, b = 79.18, c = 0.50
    # Run 40 repeated simulations over 28-day periods using Common Random Numbers (CRN)
    summary_df, traj_df, runs_df = run_benchmark_experiments(
        initial_leader_price=1.38,
        initial_follower_price=1.37,
        cost=0.83,
        a=165.39,
        b=79.18,
        c=0.50,
        competitor_true_behavior="adaptive",
        n_periods=28,
        n_simulations=40,
        rule_behavior="match",
        master_seed=42,
    )

    print("\nBenchmark Summary Results:")
    print(summary_df.to_string(index=False))

    print("\nHead of Simulation Runs:")
    print(runs_df.head(10).to_string(index=False))

    # Verify figures generate without error
    fig1 = plot_cumulative_profit(traj_df)
    fig2 = plot_strategy_comparison_bars(summary_df)
    fig3 = plot_price_trajectories(traj_df)
    fig4 = plot_profit_distribution_box(runs_df)
    fig5 = plot_revenue_comparison(traj_df)
    fig6 = plot_sales_comparison(traj_df)
    fig7 = plot_improvement_distribution(runs_df)

    print("\nAll Plotly figures generated successfully!")
    print("Experiment artifacts successfully written to data/processed/experiments/")


if __name__ == "__main__":
    main()
