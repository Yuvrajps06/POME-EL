"""End-to-End Pipeline Execution Script for PricePilot.

Executes all pipeline stages sequentially:
1. Data Preprocessing & Feature Engineering
2. Exploratory Data Analysis & Market KPIs
3. Demand Modeling, Elasticity Estimation & Linear Calibration
4. Competitor Response Model Training & Evaluation
5. Multi-Period Strategy Comparison Simulations
6. Sensitivity Analysis Sweeps & Cache Generation
"""

import sys
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import load_m5_data
from src.data.preprocessing import preprocess_m5
from src.data.feature_engineering import generate_features, chronological_split
from src.utils.config import (
    DATA_PROCESSED_DIR,
    DATA_RAW_DIR,
    DATA_SAMPLE_DIR,
    SELECTED_ITEMS,
    SELECTED_STORES,
)


def run_pipeline(force_reprocess_data: bool = False) -> None:
    """Run full PricePilot pipeline end-to-end."""
    start_time = time.time()
    print("=" * 70)
    print("PricePilot: End-to-End Pipeline Execution")
    print("=" * 70)

    # Stage 1: Data Pipeline
    print("\n[Stage 1/6] Data Ingestion & Preprocessing...")
    processed_parquet = DATA_PROCESSED_DIR / "m5_processed.parquet"

    if not processed_parquet.exists() or force_reprocess_data:
        print("  Loading raw M5 datasets...")
        sales, calendar, prices = load_m5_data(DATA_RAW_DIR / "m5")
        print("  Preprocessing selected subset...")
        clean_df = preprocess_m5(sales, calendar, prices, SELECTED_STORES, SELECTED_ITEMS)
        print("  Generating engineered features (lags, rolling stats, calendar)...")
        final_df = generate_features(clean_df)

        DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        DATA_SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
        final_df.to_parquet(processed_parquet, index=False)
        final_df.to_csv(DATA_PROCESSED_DIR / "m5_processed.csv", index=False)
        final_df.head(100).to_csv(DATA_SAMPLE_DIR / "m5_sample.csv", index=False)
        print(f"  Saved processed dataset ({len(final_df):,} rows) to {processed_parquet}")
    else:
        print(f"  Processed dataset found at {processed_parquet}. Skipping raw re-ingestion.")

    # Stage 2: EDA & Market Intelligence
    print("\n[Stage 2/6] Exploratory Data Analysis & Market KPIs...")
    from notebooks.run_eda import main as run_eda
    run_eda()

    # Stage 3: Demand Modeling
    print("\n[Stage 3/6] Demand Modeling & Elasticity Estimation...")
    from notebooks.run_demand_modeling import main as run_demand
    run_demand()

    # Stage 4: Competitor Response Modeling
    print("\n[Stage 4/6] Competitor Response Modeling & Calibration...")
    from notebooks.run_competitor_modeling import main as run_competitor
    run_competitor()

    # Stage 5: Strategy Comparison Simulation
    print("\n[Stage 5/6] Multi-Period Strategy Comparison (40 runs, 28 days, CRN)...")
    from notebooks.run_strategy_comparison import main as run_strategy_comparison
    run_strategy_comparison()

    # Stage 6: Sensitivity Analysis Sweeps
    print("\n[Stage 6/6] Sensitivity Analysis Sweeps...")
    from notebooks.run_sensitivity_analysis import main as run_sensitivity_analysis
    run_sensitivity_analysis()

    total_elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"Pipeline executed successfully in {total_elapsed:.1f} seconds!")
    print("All models, precomputed benchmarks, and sensitivity caches are ready.")
    print("=" * 70)


if __name__ == "__main__":
    force_flag = "--force" in sys.argv
    run_pipeline(force_reprocess_data=force_flag)
