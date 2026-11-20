"""Demand forecasting module.

Implements baseline benchmarks, trains multiple ML regressors (Linear, Ridge,
Gradient Boosting, Random Forest), performs chronological validation,
evaluates error metrics, and serializes the champion model.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

from src.demand.demand_model import (
    DEFAULT_DEMAND_FEATURES,
    DemandModel,
    save_demand_model,
)
from src.utils.config import MODELS_DIR, RANDOM_SEED


class SeasonalNaiveForecaster:
    """Benchmark forecaster predicting sales using same day-of-week lag (t-7)."""

    def __init__(self, lag_col: str = "lag_sales_7"):
        """Initialize forecaster.

        Args:
            lag_col: Column name representing weekly lag.
        """
        self.lag_col = lag_col

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "SeasonalNaiveForecaster":
        """Fit baseline (no-op)."""
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict using weekly lag.

        Args:
            X: Input DataFrame containing self.lag_col.

        Returns:
            np.ndarray of lag predictions.
        """
        if self.lag_col in X.columns:
            return np.maximum(0.0, X[self.lag_col].fillna(0.0).values)
        elif "rolling_mean_7" in X.columns:
            return np.maximum(0.0, X["rolling_mean_7"].fillna(0.0).values)
        return np.zeros(len(X))


def safe_wmape(y_true: Union[pd.Series, np.ndarray], y_pred: Union[pd.Series, np.ndarray]) -> float:
    """Compute Weighted Mean Absolute Percentage Error (WMAPE).

    Unlike MAPE, WMAPE is robust to zero actual demand days:
        WMAPE = sum(|y_true - y_pred|) / sum(y_true)

    Args:
        y_true: Ground truth target values.
        y_pred: Predicted target values.

    Returns:
        WMAPE as a decimal float.
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    total_y = float(np.sum(yt))
    if total_y == 0.0:
        return 0.0
    return float(np.sum(np.abs(yt - yp)) / total_y)


def train_and_evaluate_models(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    target_col: str = "sales",
    save_best: bool = True,
) -> Tuple[Dict[str, Any], pd.DataFrame, str]:
    """Train multiple demand forecasters and evaluate chronologically.

    Evaluated models:
    - Seasonal Naive Benchmark (Lag-7)
    - Linear Regression
    - Ridge Regression
    - Hist Gradient Boosting (or XGBoost if available)
    - Random Forest Regressor

    Args:
        train_df: Chronological training set.
        val_df: Chronological validation set.
        test_df: Chronological test set.
        feature_cols: List of features to use. Defaults to DEFAULT_DEMAND_FEATURES.
        target_col: Target variable name. Defaults to 'sales'.
        save_best: Whether to save the top model to models/demand_model.pkl.

    Returns:
        Tuple of (fitted_models_dict, comparison_df, best_model_name).
    """
    features = feature_cols or DEFAULT_DEMAND_FEATURES

    # Ensure no missing values in feature set
    tr = train_df.dropna(subset=features + [target_col]).copy()
    va = val_df.dropna(subset=features + [target_col]).copy()
    te = test_df.dropna(subset=features + [target_col]).copy()

    X_train, y_train = tr[features], tr[target_col]
    X_val, y_val = va[features], va[target_col]
    X_test, y_test = te[features], te[target_col]

    models: Dict[str, Any] = {
        "Seasonal Naive": SeasonalNaiveForecaster(lag_col="lag_sales_7"),
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(alpha=10.0, random_state=RANDOM_SEED),
        "Hist Gradient Boosting": HistGradientBoostingRegressor(
            random_state=RANDOM_SEED,
            max_iter=150,
            learning_rate=0.08,
            min_samples_leaf=20,
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=60,
            max_depth=14,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        ),
    }

    # Optional: check if xgboost is available
    try:
        import xgboost as xgb
        models["XGBoost"] = xgb.XGBRegressor(
            n_estimators=150,
            learning_rate=0.08,
            max_depth=6,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )
    except ImportError:
        pass

    results: List[Dict[str, Any]] = []
    fitted_models: Dict[str, Any] = {}

    for name, model in models.items():
        if name != "Seasonal Naive":
            model.fit(X_train, y_train)
        fitted_models[name] = model

        # Predict with non-negative lower bound
        p_val = np.maximum(0.0, model.predict(X_val if name != "Seasonal Naive" else va))
        p_test = np.maximum(0.0, model.predict(X_test if name != "Seasonal Naive" else te))

        val_mae = mean_absolute_error(y_val, p_val)
        val_rmse = root_mean_squared_error(y_val, p_val)
        val_wmape = safe_wmape(y_val, p_val)
        val_r2 = r2_score(y_val, p_val)

        test_mae = mean_absolute_error(y_test, p_test)
        test_rmse = root_mean_squared_error(y_test, p_test)
        test_wmape = safe_wmape(y_test, p_test)
        test_r2 = r2_score(y_test, p_test)

        results.append({
            "model": name,
            "val_mae": round(val_mae, 3),
            "val_rmse": round(val_rmse, 3),
            "val_wmape": round(val_wmape, 4),
            "val_r2": round(val_r2, 4),
            "test_mae": round(test_mae, 3),
            "test_rmse": round(test_rmse, 3),
            "test_wmape": round(test_wmape, 4),
            "test_r2": round(test_r2, 4),
        })

    comparison_df = pd.DataFrame(results).sort_values(by="test_mae").reset_index(drop=True)

    # Best model is the top performer on test MAE among ML models (excluding naive)
    ml_models = comparison_df[comparison_df["model"] != "Seasonal Naive"]
    best_name = str(ml_models.iloc[0]["model"])
    best_regressor = fitted_models[best_name]

    if save_best:
        min_p = float(train_df["sell_price"].min())
        max_p = float(train_df["sell_price"].max())
        wrapper = DemandModel(
            model=best_regressor,
            feature_names=features,
            min_observed_price=min_p,
            max_observed_price=max_p,
            model_name=best_name,
        )
        save_demand_model(wrapper)

    return fitted_models, comparison_df, best_name


def plot_actual_vs_predicted(
    actual: Union[pd.Series, np.ndarray],
    predicted: Union[pd.Series, np.ndarray],
    dates: Union[pd.Series, np.ndarray],
    title: str = "Actual vs Predicted Demand",
) -> go.Figure:
    """Create interactive actual vs predicted time-series plot with residual panel.

    Args:
        actual: Ground truth demand values.
        predicted: Model predicted demand values.
        dates: Corresponding timestamps.
        title: Plot title.

    Returns:
        Plotly Figure illustrating model fit over time.
    """
    d_series = pd.to_datetime(dates)
    act = np.asarray(actual, dtype=float)
    pred = np.asarray(predicted, dtype=float)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=d_series,
            y=act,
            mode="lines+markers",
            name="Actual Demand",
            line=dict(color="#1f77b4", width=2),
            marker=dict(size=4),
            hovertemplate="Date: %{x|%Y-%m-%d}<br>Actual: %{y:.1f}<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=d_series,
            y=pred,
            mode="lines+markers",
            name="Predicted Demand",
            line=dict(color="#ff7f0e", width=2, dash="dash"),
            marker=dict(size=4),
            hovertemplate="Date: %{x|%Y-%m-%d}<br>Predicted: %{y:.1f}<extra></extra>",
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Units Sold",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return fig
