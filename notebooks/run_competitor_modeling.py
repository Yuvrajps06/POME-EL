"""Executable demonstration script for Phase 4 competitor response module.

Trains response models, evaluates rule recovery, exercises the noisy wrapper,
and verifies the common interface get_competitor_price(...).
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from src.competitor.response_evaluation import (
    evaluate_response_classifier,
    evaluate_rule_recovery,
)
from src.competitor.response_model import (
    COMPETITOR_FEATURE_COLS,
    NoisyCompetitorModelWrapper,
    get_competitor_price,
    load_competitor_model,
    save_competitor_model,
    train_competitor_models,
)
from src.competitor.response_simulator import (
    SIMULATION_DISCLAIMER,
    generate_synthetic_competitor_dataset,
    rule_based_response,
)
from src.utils.config import COMPETITOR_RESPONSE_NOISE, DATA_PROCESSED_DIR, MODELS_DIR


def main() -> None:
    """Execute end-to-end competitor response training and evaluation."""
    print("=" * 70)
    print("PHASE 4: COMPETITOR RESPONSE MODULE")
    print("=" * 70)
    print(f"\n{SIMULATION_DISCLAIMER}\n")

    # 1. Generate synthetic competitor dataset with time ordering
    data_path = DATA_PROCESSED_DIR / "m5_processed.parquet"
    if not data_path.exists():
        data_path = DATA_PROCESSED_DIR / "m5_processed.csv"
    m5_df = pd.read_parquet(data_path) if str(data_path).endswith(".parquet") else pd.read_csv(data_path)

    print("Generating stochastic follower reaction dataset with unobserved latent factors...")
    synthetic_df = generate_synthetic_competitor_dataset(
        m5_df,
        n_samples=5000,
        behavior="adaptive",
        response_noise=COMPETITOR_RESPONSE_NOISE,
    )
    print(f"Generated {len(synthetic_df):,} synthetic interaction scenarios.")

    # Time-ordered chronological split (70% Train / 15% Val / 15% Test)
    all_dates = sorted(synthetic_df["date"].unique())
    n_dates = len(all_dates)
    val_date = all_dates[int(0.70 * n_dates)]
    test_date = all_dates[int(0.85 * n_dates)]

    train_df = synthetic_df[synthetic_df["date"] < val_date].copy().reset_index(drop=True)
    val_df = synthetic_df[(synthetic_df["date"] >= val_date) & (synthetic_df["date"] < test_date)].copy().reset_index(drop=True)
    test_df = synthetic_df[synthetic_df["date"] >= test_date].copy().reset_index(drop=True)

    print(f"Chronological Split: Train={len(train_df):,} rows (< {val_date.strftime('%Y-%m-%d')}), "
          f"Val={len(val_df):,} rows, Test={len(test_df):,} rows (>= {test_date.strftime('%Y-%m-%d')})")

    # 2. Train ML models
    print("\n--- Training ML Competitor Response Models ---")
    wrapper, metrics = train_competitor_models(train_df, val_df, save_path=None)
    print(f"Top Classifier: {metrics['best_classifier']} (Val Accuracy: {metrics['val_accuracy']:.2%})")

    # 3. Evaluate on chronological test set
    y_test_clf = test_df["target_increase"]
    pred_prices, pred_probs = wrapper.predict_response(test_df)

    clf_metrics = evaluate_response_classifier(y_test_clf, pred_probs)
    recovery_metrics = evaluate_rule_recovery(test_df, pred_probs, pred_prices)

    # Attach comprehensive benchmark metrics to wrapper and persist
    wrapper.metrics = {
        "best_classifier": metrics["best_classifier"],
        "clf_accuracy": clf_metrics["accuracy"],
        "majority_baseline": recovery_metrics["majority_baseline"],
        "ceiling_accuracy": recovery_metrics["ceiling_accuracy"],
        "relative_gap_closed_pct": recovery_metrics["relative_gap_closed_pct"],
        "clf_roc_auc": clf_metrics["roc_auc"],
        "reg_mae": recovery_metrics["price_magnitude_mae"],
        "reg_r2": recovery_metrics["price_r2"],
        "reg_rmse": recovery_metrics["price_magnitude_rmse"],
        "disclaimer": SIMULATION_DISCLAIMER,
    }
    save_competitor_model(wrapper, path=MODELS_DIR / "competitor_model.pkl")

    print("\n" + "=" * 70)
    print("COMPETITOR RESPONSE MODEL BENCHMARK (CHRONOLOGICAL TEST SET)")
    print("=" * 70)
    print(f"  Majority-Class Baseline : {recovery_metrics['majority_baseline']:.1%}")
    print(f"  Trained Model Accuracy  : {clf_metrics['accuracy']:.1%}")
    print(f"  True Rule Ceiling (Bayes): {recovery_metrics['ceiling_accuracy']:.1%}")
    print(f"  Relative Gap Closed     : {recovery_metrics['relative_gap_closed_pct']:.1f}%")
    print(f"  Classifier ROC-AUC      : {clf_metrics['roc_auc']:.3f}")
    print(f"  Classifier Brier Score  : {clf_metrics['brier_score']:.4f}")
    print(f"  Regressor Test MAE      : ${recovery_metrics['price_magnitude_mae']:.3f}")
    print(f"  Regressor Test R²       : {recovery_metrics['price_r2']:.3f}")
    print(f"  Regressor Test RMSE     : ${recovery_metrics['price_magnitude_rmse']:.3f}")
    print("=" * 70)

    # 5. Test Common Interface get_competitor_price across all 3 modes
    print("\n--- Common Interface Verification: get_competitor_price(...) ---")

    # Mode 1: Rule
    p_lead, p_foll = 1.50, 1.35
    price_rule, prob_rule = get_competitor_price(
        leader_price=p_lead,
        current_follower_price=p_foll,
        mode="rule",
        params={"behavior": "match"},
    )
    print(f"1. Mode 'rule' (match)     : P_L={p_lead:.2f}, P_F={p_foll:.2f} -> New P_F=${price_rule:.2f}, P(Inc)={prob_rule:.2f}")

    price_undercut, prob_undercut = get_competitor_price(
        leader_price=p_lead,
        current_follower_price=p_foll,
        mode="rule",
        params={"behavior": "undercut", "undercut_margin": 0.05},
    )
    print(f"   Mode 'rule' (undercut)  : P_L={p_lead:.2f}, P_F={p_foll:.2f} -> New P_F=${price_undercut:.2f}, P(Inc)={prob_undercut:.2f}")

    # Mode 2: Simulated ML
    price_ml, prob_ml = get_competitor_price(
        leader_price=p_lead,
        current_follower_price=p_foll,
        mode="simulated_ml",
        params={"model": wrapper},
    )
    print(f"2. Mode 'simulated_ml'     : P_L={p_lead:.2f}, P_F={p_foll:.2f} -> New P_F=${price_ml:.2f}, P(Inc)={prob_ml:.2f}")

    # Mode 2b: Noisy ML wrapper
    price_noisy, prob_noisy = get_competitor_price(
        leader_price=p_lead,
        current_follower_price=p_foll,
        mode="simulated_ml",
        params={"model": wrapper, "target_accuracy": 0.60},
    )
    print(f"   Mode 'simulated_ml' (60%): P_L={p_lead:.2f}, P_F={p_foll:.2f} -> New P_F=${price_noisy:.2f}, P(Inc)={prob_noisy:.2f}")

    # Mode 3: Real stub loader error handling
    try:
        get_competitor_price(leader_price=p_lead, current_follower_price=p_foll, mode="real")
    except FileNotFoundError as e:
        print(f"3. Mode 'real' (Stub Guard): Correctly raised expected error:\n   '{e}'")


if __name__ == "__main__":
    main()
