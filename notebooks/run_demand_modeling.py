"""Executable demonstration script for Phase 3 demand modeling and elasticity estimation."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

import pandas as pd
from src.data.feature_engineering import chronological_split
from src.demand.demand_model import fit_linear_approximation, load_demand_model
from src.demand.elasticity import (
    OBSERVATIONAL_CAVEAT,
    compute_per_item_elasticities,
    estimate_log_log_elasticity,
)
from src.demand.forecasting import train_and_evaluate_models
from src.utils.config import DATA_PROCESSED_DIR, MODELS_DIR, PRIMARY_MODELING_ITEM


def main() -> None:
    """Run full Phase 3 demand training, elasticity estimation, and evaluation."""
    data_path = DATA_PROCESSED_DIR / "m5_processed.parquet"
    if not data_path.exists():
        data_path = DATA_PROCESSED_DIR / "m5_processed.csv"

    print(f"Loading processed dataset from {data_path}...")
    df = pd.read_parquet(data_path) if str(data_path).endswith(".parquet") else pd.read_csv(data_path)
    df["date"] = pd.to_datetime(df["date"])

    train, val, test = chronological_split(df, test_days=28, val_days=28)
    print(f"Split sizes: Train={len(train):,}, Val={len(val):,}, Test={len(test):,}")

    # 1. Model comparison
    print("\n" + "=" * 60)
    print("DEMAND FORECASTER COMPARISON (CHRONOLOGICAL SPLIT)")
    print("=" * 60)
    fitted_models, comp_df, best_name = train_and_evaluate_models(train, val, test, save_best=True)
    print(comp_df.to_string(index=False))

    comp_csv = PROJECT_ROOT / "notebooks" / "model_comparison.csv"
    comp_df.to_csv(comp_csv, index=False)
    print(f"\nSaved model comparison to {comp_csv}")
    print(f"Champion model saved to {MODELS_DIR / 'demand_model.pkl'} ({best_name})")

    # 2. Elasticity estimation
    print("\n" + "=" * 60)
    print("LOG-LOG OBSERVATIONAL PRICE ELASTICITY ESTIMATION")
    print("=" * 60)
    primary_elas = estimate_log_log_elasticity(df, item_id=PRIMARY_MODELING_ITEM, store_id="CA_1")
    print(f"Primary Hero Item: {PRIMARY_MODELING_ITEM} (Store CA_1)")
    print(f"  Elasticity (beta_1): {primary_elas['elasticity']:.3f}")
    print(f"  Standard Error:     {primary_elas['std_error']:.3f}")
    print(f"  95% Conf Interval:  [{primary_elas['ci_lower']:.3f}, {primary_elas['ci_upper']:.3f}]")
    print(f"  p-value:            {primary_elas['p_value']:.4f}")
    print(f"  R-squared:          {primary_elas['r2']:.4f}")

    all_elas = compute_per_item_elasticities(df)
    elas_csv = PROJECT_ROOT / "notebooks" / "elasticity_estimates.csv"
    all_elas.to_csv(elas_csv, index=False)
    print(f"\nComputed elasticity across {len(all_elas)} series. Saved to {elas_csv}")

    print("\n" + OBSERVATIONAL_CAVEAT)

    # 3. Linear Approximation for Game Engine
    print("\n" + "=" * 60)
    print("LINEAR DEMAND SPECIFICATION FOR GAME ENGINE: Q = a - b*P + c*P_F + d*X")
    print("=" * 60)
    linear_specs = []
    for itm in df["item_id"].unique():
        for st in ["CA_1", "CA_2"]:
            spec = fit_linear_approximation(df, item_id=itm, store_id=st, default_c=0.50)
            linear_specs.append(spec)

    linear_df = pd.DataFrame(linear_specs)
    linear_csv = PROJECT_ROOT / "notebooks" / "linear_demand_parameters.csv"
    linear_df.to_csv(linear_csv, index=False)
    print(f"Saved linear parameters for all series to {linear_csv}")
    print(linear_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
