"""Demand modeling and price-response module.

Provides the DemandModel wrapper class, single-point demand prediction,
price-response curves with extrapolation guards, and linear demand curve
approximations (Q = a - b*P + c*P_F + d*X) for Stackelberg game theory.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import warnings

import joblib
import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.utils.config import MODELS_DIR, RANDOM_SEED

DEFAULT_DEMAND_FEATURES = [
    "sell_price",
    "price_change_pct",
    "price_rel_mean",
    "is_discounted",
    "lag_sales_1",
    "lag_sales_7",
    "lag_sales_14",
    "lag_sales_28",
    "rolling_mean_7",
    "rolling_std_7",
    "rolling_mean_28",
    "rolling_std_28",
    "day_of_week",
    "day_of_month",
    "is_weekend",
    "has_event",
    "snap_CA",
]


class DemandModel:
    """Encapsulates a trained ML demand forecaster with price response capabilities."""

    def __init__(
        self,
        model: Any,
        feature_names: List[str] = DEFAULT_DEMAND_FEATURES,
        item_id: Optional[str] = None,
        store_id: Optional[str] = None,
        min_observed_price: float = 0.50,
        max_observed_price: float = 10.0,
        model_name: str = "HistGradientBoosting",
    ):
        """Initialize DemandModel wrapper.

        Args:
            model: Trained scikit-learn or XGBoost regressor.
            feature_names: List of expected input feature columns.
            item_id: Associated item identifier.
            store_id: Associated store identifier.
            min_observed_price: Minimum price observed in training data.
            max_observed_price: Maximum price observed in training data.
            model_name: Descriptive name of the model algorithm.
        """
        self.model = model
        self.feature_names = feature_names
        self.item_id = item_id
        self.store_id = store_id
        self.min_observed_price = float(min_observed_price)
        self.max_observed_price = float(max_observed_price)
        self.model_name = model_name

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predict expected demand units with non-negative lower bound.

        Args:
            X: Input features DataFrame or array matching self.feature_names.

        Returns:
            np.ndarray of non-negative demand predictions.
        """
        if isinstance(X, pd.DataFrame):
            missing = [f for f in self.feature_names if f not in X.columns]
            if missing:
                raise ValueError(f"Missing required features: {missing}")
            input_data = X[self.feature_names]
        else:
            input_data = pd.DataFrame(X, columns=self.feature_names)

        preds = self.model.predict(input_data)
        return np.maximum(0.0, np.asarray(preds, dtype=float))

    def predict_demand(
        self,
        price: float,
        context_features: Union[Dict[str, Any], pd.Series, pd.DataFrame],
    ) -> float:
        """Predict demand units for a given price and environment context.

        Args:
            price: Evaluated selling price.
            context_features: Dictionary or Series of context features (lags, calendar, snap).

        Returns:
            Predicted non-negative demand units.
        """
        row = dict(context_features)
        row["sell_price"] = float(price)

        # Update price-dependent features if baseline price is provided
        if "item_mean_price" in row and row["item_mean_price"] > 0:
            row["price_rel_mean"] = price / row["item_mean_price"]
            row["is_discounted"] = 1 if price < row["item_mean_price"] else 0
        elif "price_rel_mean" not in row:
            row["price_rel_mean"] = 1.0
            row["is_discounted"] = 0

        if "price_change_pct" not in row:
            row["price_change_pct"] = 0.0

        # Build feature vector
        vec = [float(row.get(f, 0.0)) for f in self.feature_names]
        input_df = pd.DataFrame([vec], columns=self.feature_names)

        pred = self.predict(input_df)[0]
        return float(pred)

    def price_response_curve(
        self,
        context_features: Union[Dict[str, Any], pd.Series],
        price_grid: Optional[np.ndarray] = None,
        min_price: Optional[float] = None,
        max_price: Optional[float] = None,
        n_points: int = 30,
    ) -> pd.DataFrame:
        """Generate demand across a range of candidate prices with extrapolation warnings.

        Args:
            context_features: Baseline context features.
            price_grid: Optional explicit array of price points.
            min_price: Grid lower bound. Defaults to 0.75 * min_observed_price.
            max_price: Grid upper bound. Defaults to 1.25 * max_observed_price.
            n_points: Number of discrete evaluation points.

        Returns:
            pd.DataFrame with columns ['price', 'predicted_demand', 'is_extrapolated'].
        """
        p_min = min_price if min_price is not None else max(0.10, 0.75 * self.min_observed_price)
        p_max = max_price if max_price is not None else 1.25 * self.max_observed_price

        if price_grid is None:
            prices = np.linspace(p_min, p_max, n_points)
        else:
            prices = np.sort(price_grid)

        # Check extrapolation against historical observed range
        is_extrapolated = (prices < self.min_observed_price) | (prices > self.max_observed_price)
        if np.any(is_extrapolated):
            extrap_count = int(np.sum(is_extrapolated))
            warnings.warn(
                f"Price grid contains {extrap_count}/{len(prices)} points outside historical "
                f"observed range [${self.min_observed_price:.2f}, ${self.max_observed_price:.2f}]. "
                "Demand predictions for these points rely on model extrapolation.",
                UserWarning,
                stacklevel=2,
            )

        demands = [self.predict_demand(p, context_features) for p in prices]

        return pd.DataFrame({
            "price": prices,
            "predicted_demand": demands,
            "is_extrapolated": is_extrapolated,
        })


def fit_linear_approximation(
    df: pd.DataFrame,
    item_id: str,
    store_id: str,
    default_c: float = 0.50,
) -> Dict[str, Any]:
    """Fit linear demand approximation Q = a - b*P + d*X for the game engine.

    The Stackelberg game engine requires an empirical linear specification:
    Q_L = a - b * P_L + c * P_F + d * X
    Here, 'a' (intercept) and 'b' (own-price slope) are estimated from M5 data.
    Cross-price sensitivity 'c' is initialized as a configurable parameter since
    M5 does not contain simultaneous competitor store pricing responses.

    Args:
        df: Processed DataFrame.
        item_id: Target item ID.
        store_id: Target store ID.
        default_c: Placeholder cross-price sensitivity coefficient.

    Returns:
        Dictionary containing parameters a, b, c, r2, and metadata.
    """
    sub = df[(df["item_id"] == item_id) & (df["store_id"] == store_id)].copy()
    if len(sub) == 0:
        raise ValueError(f"No records found for item {item_id} in store {store_id}")

    dow_dummies = pd.get_dummies(sub["day_of_week"], prefix="dow", drop_first=True, dtype=float)
    month_dummies = pd.get_dummies(sub["month"], prefix="m", drop_first=True, dtype=float)

    feature_cols = ["sell_price", "has_event", "snap_CA"]
    X = pd.concat([sub[feature_cols], dow_dummies, month_dummies], axis=1).dropna()
    X = sm.add_constant(X)
    y = sub.loc[X.index, "sales"]

    ols = sm.OLS(y, X).fit()

    intercept = float(ols.params.get("const", 0.0))
    # In Q = a - b*P, the coefficient of sell_price is (-b)
    price_coef = float(ols.params.get("sell_price", 0.0))
    b_slope = -price_coef  # If price_coef is negative, b is positive
    r2 = float(ols.rsquared)

    mean_price = float(sub["sell_price"].mean())
    mean_sales = float(sub["sales"].mean())

    # Guarantee a reasonable structural own-price response for game theory:
    # If unregularized OLS yielded negative b (upward-sloping due to demand expansion),
    # calibrate b from average price and velocity so b > 0.
    if b_slope <= 0.0:
        warnings.warn(
            f"Item {item_id} in {store_id} exhibited empirical positive price slope (coef={price_coef:.3f}). "
            "Imposing structural regularized b based on unit elasticity for game theory engine.",
            UserWarning,
            stacklevel=2,
        )
        b_slope = max(1.0, mean_sales / max(mean_price, 0.1))
        intercept = mean_sales + (b_slope * mean_price)

    return {
        "item_id": item_id,
        "store_id": store_id,
        "a": round(intercept, 4),
        "b": round(b_slope, 4),
        "c": round(default_c, 4),
        "r2": round(r2, 4),
        "mean_price": round(mean_price, 2),
        "mean_sales": round(mean_sales, 2),
        "min_price": round(float(sub["sell_price"].min()), 2),
        "max_price": round(float(sub["sell_price"].max()), 2),
        "n_obs": len(sub),
    }


def save_demand_model(
    wrapper: DemandModel,
    path: Optional[Union[str, Path]] = None,
) -> Path:
    """Serialize DemandModel to disk using joblib.

    Args:
        wrapper: DemandModel instance.
        path: Destination filepath. Defaults to MODELS_DIR / 'demand_model.pkl'.

    Returns:
        Path of saved model file.
    """
    file_path = Path(path) if path else MODELS_DIR / "demand_model.pkl"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(wrapper, file_path)
    return file_path


def load_demand_model(
    path: Optional[Union[str, Path]] = None,
) -> DemandModel:
    """Deserialize DemandModel from disk using joblib.

    Args:
        path: Path to serialized file. Defaults to MODELS_DIR / 'demand_model.pkl'.

    Returns:
        DemandModel instance.
    """
    file_path = Path(path) if path else MODELS_DIR / "demand_model.pkl"
    if not file_path.exists():
        raise FileNotFoundError(f"Model file not found at {file_path}")
    return joblib.load(file_path)
