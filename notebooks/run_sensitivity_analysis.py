"""Script to run Phase 7 Sensitivity Analysis Sweeps and save artifacts."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.experiments.sensitivity import (
    run_all_sensitivity_sweeps,
    plot_heatmap_strength_x_accuracy,
    plot_accuracy_vs_profit_curve,
    plot_competitor_behavior_comparison,
    plot_demand_sensitivity_vs_price,
    plot_robustness_analysis,
)


def main():
    print("=" * 60)
    print("Running Phase 7: Sensitivity Analysis Sweeps")
    print("=" * 60)

    results = run_all_sensitivity_sweeps(
        n_simulations=15,
        n_periods=14,
        force_recompute=True,
    )

    print("\n1. Heatmap (Response Strength x ML Accuracy):")
    print(results["heatmap"].head(10).to_string(index=False))

    print("\n2. Accuracy vs Profit Curve (Symmetric vs Asymmetric Errors):")
    print(results["accuracy_curve"].to_string(index=False))

    print("\n3. Competitor Behaviors & Delays:")
    print(results["behaviors"].head(12).to_string(index=False))

    print("\n4. Demand Elasticity Sensitivity (Own b x Cross c):")
    print(results["demand"].head(10).to_string(index=False))

    print("\n5. Robustness (Cost x Demand Volatility):")
    print(results["robustness"].to_string(index=False))

    # Verify figures generate cleanly
    fig1 = plot_heatmap_strength_x_accuracy(results["heatmap"])
    fig2 = plot_accuracy_vs_profit_curve(results["accuracy_curve"])
    fig3 = plot_competitor_behavior_comparison(results["behaviors"])
    fig4 = plot_demand_sensitivity_vs_price(results["demand"])
    fig5 = plot_robustness_analysis(results["robustness"])

    print("\nAll Plotly sensitivity figures created successfully!")
    print("Artifacts saved to data/processed/experiments/")


if __name__ == "__main__":
    main()
