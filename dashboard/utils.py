"""Shared utilities, data caching, and model loaders for the Streamlit dashboard."""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import pandas as pd
import streamlit as st

from src.competitor.response_model import CompetitorResponseModel, load_competitor_model
from src.demand.demand_model import load_demand_model
from src.utils.config import DATA_PROCESSED_DIR, MODELS_DIR

NOTEBOOKS_DIR = Path(__file__).resolve().parent.parent / "notebooks"
EXPERIMENTS_DIR = DATA_PROCESSED_DIR / "experiments"


@st.cache_data(show_spinner="Loading M5 processed data...")
def load_processed_dataset() -> pd.DataFrame:
    """Load processed M5 dataset, preferring parquet format for speed."""
    parquet_path = DATA_PROCESSED_DIR / "m5_processed.parquet"
    csv_path = DATA_PROCESSED_DIR / "m5_processed.csv"

    if parquet_path.exists():
        df = pd.read_parquet(parquet_path)
    elif csv_path.exists():
        df = pd.read_csv(csv_path)
    else:
        st.error(f"Processed dataset not found at {parquet_path} or {csv_path}.")
        return pd.DataFrame()

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_resource(show_spinner="Loading demand model...")
def get_cached_demand_model() -> Optional[Any]:
    """Load serialized demand model gracefully."""
    try:
        return load_demand_model()
    except Exception as e:
        st.warning(f"Demand model could not be loaded: {e}")
        return None


@st.cache_resource(show_spinner="Loading competitor model...")
def get_cached_competitor_model() -> Optional[CompetitorResponseModel]:
    """Load serialized competitor model gracefully."""
    try:
        return load_competitor_model()
    except Exception as e:
        st.warning(f"Competitor response model could not be loaded: {e}")
        return None


@st.cache_data
def load_market_kpis_table() -> pd.DataFrame:
    """Load precomputed market intelligence KPIs."""
    path = NOTEBOOKS_DIR / "market_kpis.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_data
def load_model_comparison_table() -> pd.DataFrame:
    """Load precomputed demand forecasting model evaluation table."""
    path = NOTEBOOKS_DIR / "model_comparison.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_data
def load_elasticity_table() -> pd.DataFrame:
    """Load precomputed log-log elasticity estimates."""
    path = NOTEBOOKS_DIR / "elasticity_estimates.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_data
def load_linear_demand_parameters() -> pd.DataFrame:
    """Load structural linear demand parameters (a, b)."""
    path = NOTEBOOKS_DIR / "linear_demand_parameters.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_data
def load_strategy_comparison_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load precomputed strategy comparison benchmark datasets."""
    summary_path = EXPERIMENTS_DIR / "strategy_comparison_summary.csv"
    traj_path = EXPERIMENTS_DIR / "strategy_trajectories.csv"
    runs_path = EXPERIMENTS_DIR / "strategy_simulation_runs.csv"

    df_summary = pd.read_csv(summary_path) if summary_path.exists() else pd.DataFrame()
    df_traj = pd.read_csv(traj_path) if traj_path.exists() else pd.DataFrame()
    df_runs = pd.read_csv(runs_path) if runs_path.exists() else pd.DataFrame()

    return df_summary, df_traj, df_runs


@st.cache_data
def load_sensitivity_datasets() -> Dict[str, pd.DataFrame]:
    """Load precomputed Phase 7 sensitivity analysis sweep tables."""
    files = {
        "heatmap": EXPERIMENTS_DIR / "sensitivity_heatmap_strength_accuracy.csv",
        "accuracy_curve": EXPERIMENTS_DIR / "sensitivity_accuracy_curve.csv",
        "behaviors": EXPERIMENTS_DIR / "sensitivity_competitor_behaviors.csv",
        "demand": EXPERIMENTS_DIR / "sensitivity_demand_slopes.csv",
        "robustness": EXPERIMENTS_DIR / "sensitivity_robustness.csv",
    }
    dfs = {}
    for k, p in files.items():
        dfs[k] = pd.read_csv(p) if p.exists() else pd.DataFrame()
    return dfs
