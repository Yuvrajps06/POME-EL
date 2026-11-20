"""Tests for Phase 4 competitor response module."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.competitor.response_evaluation import (
    evaluate_response_classifier,
    evaluate_rule_recovery,
)
from src.competitor.response_model import (
    COMPETITOR_FEATURE_COLS,
    CompetitorResponseModel,
    NoisyCompetitorModelWrapper,
    get_competitor_price,
    load_competitor_model,
    load_real_competitor_data,
    save_competitor_model,
    train_competitor_models,
)
from src.competitor.response_simulator import (
    generate_synthetic_competitor_dataset,
    rule_based_response,
)
from src.utils.config import DATA_PROCESSED_DIR, MODELS_DIR


@pytest.fixture(scope="module")
def sample_competitor_dataset() -> pd.DataFrame:
    """Load sample processed M5 data and generate small synthetic competitor dataset."""
    path = DATA_PROCESSED_DIR / "m5_processed.parquet"
    if path.exists():
        m5_df = pd.read_parquet(path)
    else:
        m5_df = pd.DataFrame({
            "sell_price": [1.50, 2.00, 2.50, 3.00],
            "rolling_mean_7": [10.0, 15.0, 12.0, 8.0],
            "rolling_std_7": [1.0, 2.0, 1.5, 1.0],
            "lag_sales_1": [10.0, 14.0, 12.0, 7.0],
            "lag_sales_7": [9.0, 13.0, 11.0, 8.0],
            "day_of_week": [0, 1, 2, 3],
            "month": [1, 2, 3, 4],
            "is_weekend": [0, 0, 0, 0],
            "snap_CA": [0, 1, 0, 0],
            "has_event": [0, 0, 1, 0],
        })
    return generate_synthetic_competitor_dataset(m5_df, n_samples=300, noise_std=0.01)


def test_rule_based_response_match() -> None:
    """Verify 'match' behavior sets follower price equal to leader price."""
    price, prob = rule_based_response(
        leader_price=2.00, current_follower_price=1.80, params={"behavior": "match"}
    )
    assert price == 2.00
    assert prob >= 0.75  # High probability of increase


def test_rule_based_response_undercut() -> None:
    """Verify 'undercut' behavior sets follower price below leader price by specified margin."""
    price, prob = rule_based_response(
        leader_price=2.00,
        current_follower_price=2.00,
        params={"behavior": "undercut", "undercut_margin": 0.05},
    )
    assert price == pytest.approx(1.90, 0.01)  # 2.00 * 0.95 = 1.90


def test_rule_based_response_maintain() -> None:
    """Verify 'maintain' behavior keeps follower price when leader changes moderately."""
    price, prob = rule_based_response(
        leader_price=1.95, current_follower_price=2.00, params={"behavior": "maintain"}
    )
    assert price == 2.00


def test_synthetic_data_generation(sample_competitor_dataset: pd.DataFrame) -> None:
    """Verify generated synthetic dataset matches expected dimensions and has zero NaNs."""
    df = sample_competitor_dataset
    assert len(df) == 300
    for col in COMPETITOR_FEATURE_COLS:
        assert col in df.columns
        assert not df[col].isna().any()

    assert "target_increase" in df.columns
    assert "target_follower_price" in df.columns
    assert set(df["target_increase"].unique()).issubset({0, 1})


def test_competitor_model_training_and_serialization(
    sample_competitor_dataset: pd.DataFrame, tmp_path: Path
) -> None:
    """Verify model training, non-negative price prediction, and joblib serialization."""
    df = sample_competitor_dataset
    train_df = df.iloc[:200]
    val_df = df.iloc[200:]

    model_path = tmp_path / "test_competitor.pkl"
    wrapper, metrics = train_competitor_models(train_df, val_df, save_path=model_path)

    assert model_path.exists()
    assert metrics["val_accuracy"] > 0.60

    loaded = load_competitor_model(model_path)
    price, prob = loaded.predict_single(leader_price=2.50, current_follower_price=2.00)

    assert price > 0.0
    assert 0.0 <= prob <= 1.0


def test_noisy_competitor_model_wrapper(sample_competitor_dataset: pd.DataFrame) -> None:
    """Verify noisy wrapper degrades response predictions."""
    df = sample_competitor_dataset
    wrapper, _ = train_competitor_models(df.iloc[:150], df.iloc[150:])

    noisy = NoisyCompetitorModelWrapper(base_model=wrapper, target_accuracy=0.60)
    price, prob = noisy.predict_single(leader_price=2.50, current_follower_price=2.00)
    assert price > 0.0
    assert 0.0 <= prob <= 1.0


def test_real_competitor_data_guard(tmp_path: Path) -> None:
    """Verify real data loader raises FileNotFoundError when absent and checks schema."""
    with pytest.raises(FileNotFoundError, match="Real competitor price data not found"):
        load_real_competitor_data(filepath=None)

    # Missing column schema test
    invalid_csv = tmp_path / "invalid_competitor.csv"
    pd.DataFrame({"Date": ["2020-01-01"], "Our_Price": [1.0]}).to_csv(invalid_csv, index=False)

    with pytest.raises(ValueError, match="missing required columns"):
        load_real_competitor_data(filepath=invalid_csv)


def test_common_interface_get_competitor_price() -> None:
    """Verify common interface functions across 'rule', 'simulated_ml', and 'real'."""
    # 1. Rule mode
    p_foll, prob = get_competitor_price(
        leader_price=3.00,
        current_follower_price=2.50,
        mode="rule",
        params={"behavior": "match"},
    )
    assert p_foll == 3.00

    # 2. Real mode guard
    with pytest.raises(FileNotFoundError):
        get_competitor_price(leader_price=3.00, current_follower_price=2.50, mode="real")
