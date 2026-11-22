"""Competitor response model evaluation and rule-recovery analysis module.

Evaluates classification metrics (Accuracy, Precision, Recall, F1, Brier score),
confusion matrices, probability calibration, and empirical recovery of the known
ground-truth behavioral simulation rules.
"""

from typing import Any, Dict, List, Tuple, Union
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    root_mean_squared_error,
)

from src.competitor.response_simulator import SIMULATION_DISCLAIMER


def evaluate_response_classifier(
    y_true: Union[pd.Series, np.ndarray],
    y_prob: Union[pd.Series, np.ndarray],
    threshold: float = 0.50,
) -> Dict[str, Any]:
    """Calculate comprehensive classification metrics for competitor direction prediction.

    Args:
        y_true: Binary ground-truth action (1 = Increase, 0 = Maintain/Decrease).
        y_prob: Predicted probability of price increase.
        threshold: Decision boundary threshold.

    Returns:
        Dict of evaluation metrics with simulation labels.
    """
    yt = np.asarray(y_true, dtype=int)
    yp = np.asarray(y_prob, dtype=float)
    y_pred = (yp >= threshold).astype(int)

    acc = float(accuracy_score(yt, y_pred))
    prec = float(precision_score(yt, y_pred, zero_division=0))
    rec = float(recall_score(yt, y_pred, zero_division=0))
    f1 = float(f1_score(yt, y_pred, zero_division=0))
    auc = float(roc_auc_score(yt, yp)) if len(np.unique(yt)) > 1 else 0.50
    brier = float(brier_score_loss(yt, yp))
    cm = confusion_matrix(yt, y_pred).tolist()

    return {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "roc_auc": round(auc, 4),
        "brier_score": round(brier, 4),
        "confusion_matrix": cm,
        "provenance_label": "SIMULATED DATA (not real-world accuracy)",
        "disclaimer": SIMULATION_DISCLAIMER,
    }


def evaluate_rule_recovery(
    test_df: pd.DataFrame,
    predicted_probs: np.ndarray,
    predicted_prices: np.ndarray,
) -> Dict[str, Any]:
    """Assess how accurately the ML models recovered the synthetic ground-truth behavioral rule.

    Calculates:
    - Directional recovery accuracy (Match between ML binary choice and ground truth action)
    - Probability calibration error (MAE between ML predicted probability and true logistic probability)
    - Price magnitude error (MAE between ML predicted price and true rule price)

    Args:
        test_df: Synthetic test DataFrame containing ground truth columns.
        predicted_probs: Model predicted increase probabilities.
        predicted_prices: Model predicted follower prices.

    Returns:
        Dict summarizing rule recovery fidelity.
    """
    true_action = test_df["target_increase"].values
    true_prob = test_df["ground_truth_prob"].values
    true_price = test_df["target_follower_price"].values

    pred_action = (predicted_probs >= 0.50).astype(int)
    rule_match_rate = float(np.mean(pred_action == true_action))
    prob_mae = float(mean_absolute_error(true_prob, predicted_probs))
    price_mae = float(mean_absolute_error(true_price, predicted_prices))
    price_rmse = float(root_mean_squared_error(true_price, predicted_prices))
    price_r2 = float(r2_score(true_price, predicted_prices))

    majority_baseline = float(max(np.mean(true_action == 1), np.mean(true_action == 0)))
    if "bayes_optimal_pred" in test_df.columns:
        ceiling_accuracy = float(np.mean(test_df["bayes_optimal_pred"].values == true_action))
    else:
        ceiling_accuracy = float(np.mean((true_prob >= 0.50) == true_action))

    gap_closed = float((rule_match_rate - majority_baseline) / max(1e-4, ceiling_accuracy - majority_baseline) * 100.0)

    return {
        "rule_recovery_rate": round(rule_match_rate, 4),
        "majority_baseline": round(majority_baseline, 4),
        "ceiling_accuracy": round(ceiling_accuracy, 4),
        "relative_gap_closed_pct": round(gap_closed, 2),
        "prob_recovery_mae": round(prob_mae, 4),
        "price_magnitude_mae": round(price_mae, 4),
        "price_magnitude_rmse": round(price_rmse, 4),
        "price_r2": round(price_r2, 4),
        "provenance_label": "SIMULATED DATA (not real-world accuracy)",
    }


def plot_confusion_matrix(
    cm: List[List[int]],
    labels: Tuple[str, str] = ("Maintain/Cut", "Increase"),
) -> go.Figure:
    """Plot interactive Plotly heatmap for confusion matrix.

    Args:
        cm: 2x2 list or array confusion matrix.
        labels: Class names.

    Returns:
        Plotly Figure heatmap.
    """
    z = np.array(cm)
    fig = px.imshow(
        z,
        text_auto=True,
        x=list(labels),
        y=list(labels),
        labels=dict(x="Predicted Follower Action", y="True Follower Action"),
        color_continuous_scale="Blues",
        title="Competitor Response Confusion Matrix (Simulated Data; Not Real-World)",
    )
    fig.update_layout(template="plotly_white")
    return fig


def plot_calibration_curve(
    y_true: Union[pd.Series, np.ndarray],
    y_prob: Union[pd.Series, np.ndarray],
    n_bins: int = 10,
) -> go.Figure:
    """Plot probability calibration curve (reliability diagram).

    Args:
        y_true: Ground truth binary outcomes.
        y_prob: Predicted probabilities.
        n_bins: Discretization bin count.

    Returns:
        Plotly Figure comparing empirical frequency vs predicted confidence.
    """
    yt = np.asarray(y_true, dtype=int)
    yp = np.asarray(y_prob, dtype=float)

    prob_true, prob_pred = calibration_curve(yt, yp, n_bins=n_bins, strategy="uniform")

    fig = go.Figure()

    # Model curve
    fig.add_trace(
        go.Scatter(
            x=prob_pred,
            y=prob_true,
            mode="lines+markers",
            name="Model Calibration",
            line=dict(color="#1f77b4", width=2.5),
            marker=dict(size=7),
        )
    )

    # Perfectly calibrated diagonal reference
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Perfect Calibration",
            line=dict(color="gray", dash="dash"),
        )
    )

    fig.update_layout(
        title="Follower Response Probability Calibration (Simulated Data; Not Real-World)",
        xaxis_title="Predicted Probability of Price Increase",
        yaxis_title="Empirical Fraction of Positives",
        template="plotly_white",
    )
    return fig


def plot_rule_recovery(
    leader_pct_changes: np.ndarray,
    true_probs: np.ndarray,
    predicted_probs: np.ndarray,
) -> go.Figure:
    """Plot true behavioral rule response curve vs ML predicted response.

    Visualizes how closely the ML model learned the ground-truth reaction function
    across leader price changes.

    Args:
        leader_pct_changes: Array of leader price changes %.
        true_probs: Ground truth probability of follower increase.
        predicted_probs: Model predicted probability of increase.

    Returns:
        Plotly Figure comparing ground-truth rule to ML recovery.
    """
    order = np.argsort(leader_pct_changes)
    x = leader_pct_changes[order] * 100.0
    y_true = true_probs[order]
    y_pred = predicted_probs[order]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x,
            y=y_true,
            mode="lines",
            name="Ground Truth Rule",
            line=dict(color="#2ca02c", width=3),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=x,
            y=y_pred,
            mode="markers",
            name="ML Model Recovery",
            marker=dict(color="#1f77b4", size=5, opacity=0.4),
        )
    )

    fig.update_layout(
        title="Rule Recovery Fidelity: True vs ML Follower Reaction (Simulated Data; Not Real-World)",
        xaxis_title="Leader Price Change (%)",
        yaxis_title="Probability of Follower Price Increase",
        template="plotly_white",
    )
    return fig
