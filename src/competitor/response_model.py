"""Competitor response modeling and unified decision support interface.

Implements ML response classifiers, magnitude regressors, a noisy ML wrapper
for sensitivity experiments, a stub loader for real competitor data, and the
common interface: get_competitor_price(...).
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Ridge

from src.competitor.response_simulator import (
    SIMULATION_DISCLAIMER,
    rule_based_response,
)
from src.utils.config import MODELS_DIR, RANDOM_SEED

COMPETITOR_FEATURE_COLS = [
    "leader_price",
    "leader_prev_price",
    "leader_price_change_pct",
    "follower_price",
    "price_gap",
    "price_ratio",
    "recent_demand",
    "sales_volatility",
    "demand_trend",
    "day_of_week",
    "month",
    "is_weekend",
    "snap_CA",
    "has_event",
]

REAL_DATA_REQUIRED_SCHEMA = [
    "Date",
    "Our_Price",
    "Competitor_Price",
    "Our_Sales",
    "Competitor_Sales",
]


class CompetitorResponseModel:
    """Trained ML container predicting follower direction and price response magnitude."""

    def __init__(
        self,
        classifier: Any,
        regressor: Any,
        feature_names: List[str] = COMPETITOR_FEATURE_COLS,
        classifier_name: str = "HistGradientBoostingClassifier",
        regressor_name: str = "HistGradientBoostingRegressor",
    ):
        """Initialize CompetitorResponseModel.

        Args:
            classifier: Model predicting P(Increase).
            regressor: Model predicting follower new price.
            feature_names: Expected feature columns.
            classifier_name: Algorithm name for the classifier.
            regressor_name: Algorithm name for the regressor.
        """
        self.classifier = classifier
        self.regressor = regressor
        self.feature_names = feature_names
        self.classifier_name = classifier_name
        self.regressor_name = regressor_name
        self.disclaimer = SIMULATION_DISCLAIMER

    def predict_response(
        self,
        features_df: pd.DataFrame,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Predict follower prices and increase probabilities for batch inputs.

        Args:
            features_df: DataFrame matching self.feature_names.

        Returns:
            Tuple of (predicted_prices, probabilities_of_increase).
        """
        X = features_df[self.feature_names]
        probs = self.classifier.predict_proba(X)[:, 1]
        prices = self.regressor.predict(X)
        return np.maximum(0.10, prices), probs

    def predict_single(
        self,
        leader_price: float,
        current_follower_price: float,
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[float, float]:
        """Predict single follower reaction to leader pricing decision.

        Args:
            leader_price: Leader's proposed price.
            current_follower_price: Follower's existing price.
            context: Environment features (recent sales, calendar).

        Returns:
            Tuple of (follower_price: float, probability_of_increase: float).
        """
        ctx = context or {}
        leader_prev = float(ctx.get("leader_prev_price", current_follower_price))
        leader_pct = (leader_price - leader_prev) / max(0.01, leader_prev)

        row = {
            "leader_price": float(leader_price),
            "leader_prev_price": leader_prev,
            "leader_price_change_pct": float(leader_pct),
            "follower_price": float(current_follower_price),
            "price_gap": float(leader_price - current_follower_price),
            "price_ratio": float(leader_price / max(0.01, current_follower_price)),
            "recent_demand": float(ctx.get("recent_demand", 10.0)),
            "sales_volatility": float(ctx.get("sales_volatility", 2.0)),
            "demand_trend": float(ctx.get("demand_trend", 0.0)),
            "day_of_week": int(ctx.get("day_of_week", 2)),
            "month": int(ctx.get("month", 5)),
            "is_weekend": int(ctx.get("is_weekend", 0)),
            "snap_CA": int(ctx.get("snap_CA", 0)),
            "has_event": int(ctx.get("has_event", 0)),
        }

        df_in = pd.DataFrame([row])
        prices, probs = self.predict_response(df_in)
        return round(float(prices[0]), 2), round(float(probs[0]), 4)


class NoisyCompetitorModelWrapper:
    """Wrapper that degrades an ML response predictor to a target accuracy level (50-90%).

    Used specifically in Phase 7 sensitivity experiments to assess Stackelberg
    profit gains as follower prediction quality varies from random coin-flip (50%)
    to near-perfect foresight (90%+).
    """

    def __init__(
        self,
        base_model: CompetitorResponseModel,
        target_accuracy: float = 0.70,
        error_mode: str = "symmetric",
        random_seed: int = RANDOM_SEED,
    ):
        """Initialize noisy wrapper.

        Args:
            base_model: Trained CompetitorResponseModel.
            target_accuracy: Controlled target accuracy in range [0.50, 0.99].
            error_mode: Error asymmetry mode:
                - 'symmetric': Uniform direction flip on error.
                - 'optimistic': Biased towards predicting competitor matches/raises upwards.
                - 'pessimistic': Biased towards predicting competitor maintains/undercuts downwards.
            random_seed: Seed for stochastic degradation.
        """
        self.base_model = base_model
        self.target_accuracy = float(np.clip(target_accuracy, 0.50, 0.99))
        self.error_mode = str(error_mode).lower()
        self.rng = np.random.RandomState(random_seed)

    def predict_response(
        self,
        df_features: pd.DataFrame,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Batch predict competitor responses with controlled degradation.

        Args:
            df_features: Feature DataFrame containing 'follower_price', etc.

        Returns:
            Tuple of (predicted_prices_array, predicted_probabilities_array).
        """
        base_prices, base_probs = self.base_model.predict_response(df_features)
        follower_prev = df_features["follower_price"].to_numpy(dtype=float)
        n = len(df_features)

        draws = self.rng.uniform(0.0, 1.0, size=n)
        error_mask = draws > self.target_accuracy

        noisy_prices = base_prices.copy()
        noisy_probs = base_probs.copy()

        if np.any(error_mask):
            if self.error_mode == "optimistic":
                # Optimistic error: model predicts competitor raises/matches even if ground truth is undercut/maintain
                noisy_prices[error_mask] = np.maximum(
                    base_prices[error_mask],
                    follower_prev[error_mask] * 1.05,
                )
                noisy_probs[error_mask] = np.clip(base_probs[error_mask] + 0.35, 0.01, 0.99)
            elif self.error_mode == "pessimistic":
                # Pessimistic error: model predicts competitor undercuts even if competitor raises
                noisy_prices[error_mask] = np.minimum(
                    base_prices[error_mask],
                    follower_prev[error_mask] * 0.95,
                )
                noisy_probs[error_mask] = np.clip(base_probs[error_mask] - 0.35, 0.01, 0.99)
            else:
                # Symmetric: flip deviation direction
                delta = base_prices[error_mask] - follower_prev[error_mask]
                noisy_prices[error_mask] = np.maximum(0.10, follower_prev[error_mask] - delta)
                noisy_probs[error_mask] = 1.0 - base_probs[error_mask]

        return np.round(noisy_prices, 2), np.round(noisy_probs, 4)

    def predict_single(
        self,
        leader_price: float,
        current_follower_price: float,
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[float, float]:
        """Predict with controlled error injection to achieve target accuracy."""
        pred_price, pred_prob = self.base_model.predict_single(
            leader_price, current_follower_price, context
        )

        # Inject classification flip with error rate (1 - target_accuracy)
        if self.rng.uniform(0.0, 1.0) > self.target_accuracy:
            if self.error_mode == "optimistic":
                noisy_price = max(pred_price, current_follower_price * 1.05)
                noisy_prob = min(0.99, pred_prob + 0.35)
            elif self.error_mode == "pessimistic":
                noisy_price = min(pred_price, current_follower_price * 0.95)
                noisy_prob = max(0.01, pred_prob - 0.35)
            else:
                noisy_prob = 1.0 - pred_prob
                delta = pred_price - current_follower_price
                noisy_price = current_follower_price - delta
            return round(float(max(0.10, noisy_price)), 2), round(float(noisy_prob), 4)

        return pred_price, pred_prob


def load_real_competitor_data(
    filepath: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """Stub loader for real external competitor datasets with schema verification.

    Expected Schema:
    - Date: Observation date (YYYY-MM-DD)
    - Our_Price: Selling price of leader firm
    - Competitor_Price: Observed selling price of competitor firm
    - Our_Sales: Leader units sold
    - Competitor_Sales: Competitor units sold

    Args:
        filepath: Path to real competitor CSV/parquet file.

    Returns:
        pd.DataFrame containing verified competitor data.

    Raises:
        FileNotFoundError: If external competitor dataset does not exist.
        ValueError: If required schema columns are missing.
    """
    if filepath is None:
        raise FileNotFoundError(
            "Real competitor price data not found. As noted in CONTEXT.md, the Walmart M5 "
            "dataset contains NO external competitor pricing. Mode 'real' requires an external "
            f"dataset with verified schema: {REAL_DATA_REQUIRED_SCHEMA}."
        )

    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Real competitor data file not found at: {path}")

    df = pd.read_csv(path)
    missing = [c for c in REAL_DATA_REQUIRED_SCHEMA if c not in df.columns]
    if missing:
        raise ValueError(f"Real competitor dataset missing required columns: {missing}")

    return df


def get_competitor_price(
    leader_price: float,
    current_follower_price: float,
    context: Optional[Dict[str, Any]] = None,
    mode: str = "rule",
    params: Optional[Dict[str, Any]] = None,
) -> Tuple[float, float]:
    """Common interface for obtaining follower competitor response.

    Modes:
    1. 'rule': Explicit rule-based reaction (match, undercut, maintain, raise, adaptive).
    2. 'simulated_ml': ML model trained on synthetic behavioral response data.
    3. 'real': Real competitor data loader (raises clear error if unsupplied).

    Args:
        leader_price: Leader's intended price.
        current_follower_price: Follower's prevailing price.
        context: Environmental covariates (demand, seasonality, SNAP).
        mode: Operation mode in {'rule', 'simulated_ml', 'real'}.
        params: Mode-specific parameters:
            - For 'rule': behavior, response_strength, undercut_margin, noise_std.
            - For 'simulated_ml': model_path, target_accuracy (for noisy degradation).
            - For 'real': filepath.

    Returns:
        Tuple of (follower_price: float, probability_of_increase: float).
    """
    p = params or {}
    mode_clean = str(mode).lower().strip()

    if mode_clean == "rule":
        return rule_based_response(
            leader_price=leader_price,
            current_follower_price=current_follower_price,
            context=context,
            params=p,
        )

    elif mode_clean == "simulated_ml":
        model_obj = p.get("model")
        if model_obj is None:
            # Load default trained competitor model
            model_path = p.get("model_path", MODELS_DIR / "competitor_model.pkl")
            model_obj = load_competitor_model(model_path)

        # Check for noisy wrapper
        target_acc = p.get("target_accuracy")
        if target_acc is not None and target_acc < 1.0:
            wrapper = NoisyCompetitorModelWrapper(
                base_model=model_obj, target_accuracy=target_acc
            )
            return wrapper.predict_single(leader_price, current_follower_price, context)

        return model_obj.predict_single(leader_price, current_follower_price, context)

    elif mode_clean == "real":
        # Stub loader verifying external schema
        filepath = p.get("filepath")
        # Raises clear documented error if missing
        _ = load_real_competitor_data(filepath)
        raise NotImplementedError("Real competitor predictive module is pending real data acquisition.")

    else:
        raise ValueError(
            f"Unknown competitor mode '{mode}'. Supported modes: 'rule', 'simulated_ml', 'real'."
        )


def train_competitor_models(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    save_path: Optional[Union[str, Path]] = None,
) -> Tuple[CompetitorResponseModel, Dict[str, Any]]:
    """Train ML classifiers and magnitude regressors on synthetic competitor dataset.

    Args:
        train_df: Synthetic training set.
        val_df: Synthetic validation set.
        feature_cols: Feature columns list.
        save_path: Path to serialize the champion model.

    Returns:
        Tuple of (CompetitorResponseModel, evaluation_metrics_dict).
    """
    features = feature_cols or COMPETITOR_FEATURE_COLS

    X_train = train_df[features].fillna(0.0)
    y_train_clf = train_df["target_increase"]
    y_train_reg = train_df["target_follower_price"]

    X_val = val_df[features].fillna(0.0)
    y_val_clf = val_df["target_increase"]
    y_val_reg = val_df["target_follower_price"]

    # Classifiers to compare
    classifiers = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_SEED),
        "Random Forest Classifier": RandomForestClassifier(
            n_estimators=100, max_depth=8, random_state=RANDOM_SEED, n_jobs=-1
        ),
        "Hist Gradient Boosting Classifier": HistGradientBoostingClassifier(
            max_iter=120, learning_rate=0.08, random_state=RANDOM_SEED
        ),
    }

    best_clf_name = ""
    best_clf_acc = -1.0
    best_clf = None

    for name, clf in classifiers.items():
        clf.fit(X_train, y_train_clf)
        acc = float(clf.score(X_val, y_val_clf))
        if acc > best_clf_acc:
            best_clf_acc = acc
            best_clf_name = name
            best_clf = clf

    # Magnitude regressor
    regressor = HistGradientBoostingRegressor(
        max_iter=120, learning_rate=0.08, random_state=RANDOM_SEED
    )
    regressor.fit(X_train, y_train_reg)

    wrapper = CompetitorResponseModel(
        classifier=best_clf,
        regressor=regressor,
        feature_names=features,
        classifier_name=best_clf_name,
        regressor_name="HistGradientBoostingRegressor",
    )

    if save_path or save_path is None:
        target_path = Path(save_path) if save_path else MODELS_DIR / "competitor_model.pkl"
        save_competitor_model(wrapper, path=target_path)

    metrics = {
        "best_classifier": best_clf_name,
        "val_accuracy": round(best_clf_acc, 4),
        "disclaimer": SIMULATION_DISCLAIMER,
    }

    return wrapper, metrics


def save_competitor_model(
    wrapper: CompetitorResponseModel,
    path: Optional[Union[str, Path]] = None,
) -> Path:
    """Serialize CompetitorResponseModel to disk with joblib.

    Args:
        wrapper: CompetitorResponseModel instance.
        path: Target file path. Defaults to MODELS_DIR / 'competitor_model.pkl'.

    Returns:
        Path of saved model.
    """
    file_path = Path(path) if path else MODELS_DIR / "competitor_model.pkl"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(wrapper, file_path)
    return file_path


def load_competitor_model(
    path: Optional[Union[str, Path]] = None,
) -> CompetitorResponseModel:
    """Load serialized CompetitorResponseModel from disk.

    Args:
        path: Path to serialized file. Defaults to MODELS_DIR / 'competitor_model.pkl'.

    Returns:
        CompetitorResponseModel instance.
    """
    file_path = Path(path) if path else MODELS_DIR / "competitor_model.pkl"
    if not file_path.exists():
        raise FileNotFoundError(f"Competitor model file not found at: {file_path}")
    return joblib.load(file_path)
