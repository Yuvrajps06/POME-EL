"""Price elasticity estimation module using log-log regression.

Fits log-log econometric specifications with calendar, event, SNAP, and
store-level controls. Reports point estimates, standard errors, p-values,
and 95% confidence intervals with explicit observational caveats.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm

OBSERVATIONAL_CAVEAT: str = (
    "Methodological Caveat: Price elasticity estimated from historical regression is "
    "observational, NOT causal. It captures historical co-movements between price and "
    "sales volume conditional on observed calendar, holiday, and SNAP controls. "
    "It does not represent a randomized counterfactual experiment and may remain "
    "confounded by unobserved demand shocks, promotional displays, or store traffic."
)


def estimate_log_log_elasticity(
    df: pd.DataFrame,
    item_id: Optional[str] = None,
    store_id: Optional[str] = None,
    controls: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Fit log-log demand model to estimate price elasticity for an item-store series.

    Specification:
        ln(Q + 1) = beta_0 + beta_1 * ln(P) + gamma * Controls + epsilon
    where beta_1 represents price elasticity (% change in demand per 1% change in price).

    Args:
        df: Processed DataFrame with log_price and log_sales.
        item_id: Filter by target item.
        store_id: Filter by target store.
        controls: Optional list of additional continuous/discrete control columns.

    Returns:
        Dict with elasticity estimate, standard error, t-stat, p-value, 95% CI, R2, and caveat.
    """
    sub = df.copy()
    if item_id:
        sub = sub[sub["item_id"] == item_id]
    if store_id:
        sub = sub[sub["store_id"] == store_id]

    if len(sub) < 30:
        raise ValueError(f"Insufficient data ({len(sub)} rows) to fit log-log regression.")

    # Categorical fixed effects
    dow_dummies = pd.get_dummies(sub["day_of_week"], prefix="dow", drop_first=True, dtype=float)
    month_dummies = pd.get_dummies(sub["month"], prefix="m", drop_first=True, dtype=float)

    # Base controls
    base_controls = ["has_event", "snap_CA"]
    if controls:
        base_controls.extend([c for c in controls if c in sub.columns and c not in base_controls])

    # Store dummy if pooled across stores
    store_dummies = pd.DataFrame(index=sub.index)
    if sub["store_id"].nunique() > 1:
        store_dummies = pd.get_dummies(sub["store_id"], prefix="store", drop_first=True, dtype=float)

    X_parts = [sub[["log_price"] + [c for c in base_controls if c in sub.columns]], dow_dummies, month_dummies]
    if not store_dummies.empty:
        X_parts.append(store_dummies)

    X = pd.concat(X_parts, axis=1).dropna()
    X = sm.add_constant(X)
    y = sub.loc[X.index, "log_sales"]

    model = sm.OLS(y, X).fit()

    if "log_price" not in model.params:
        raise ValueError("log_price coefficient could not be estimated.")

    coef = float(model.params["log_price"])
    se = float(model.bse["log_price"])
    t_stat = float(model.tvalues["log_price"])
    p_val = float(model.pvalues["log_price"])
    ci = model.conf_int().loc["log_price"].values

    return {
        "item_id": item_id or "POOLED",
        "store_id": store_id or "POOLED",
        "elasticity": round(coef, 4),
        "std_error": round(se, 4),
        "t_statistic": round(t_stat, 3),
        "p_value": round(p_val, 5),
        "ci_lower": round(float(ci[0]), 4),
        "ci_upper": round(float(ci[1]), 4),
        "r2": round(float(model.rsquared), 4),
        "n_obs": int(model.nobs),
        "is_stat_sig_5pct": bool(p_val < 0.05),
        "caveat": OBSERVATIONAL_CAVEAT,
    }


def compute_per_item_elasticities(
    df: pd.DataFrame,
    items: Optional[List[str]] = None,
    stores: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Compute observational elasticity across all item-store series.

    Args:
        df: Processed DataFrame.
        items: Optional subset of item IDs.
        stores: Optional subset of store IDs.

    Returns:
        pd.DataFrame summarizing per-item elasticity estimates and confidence intervals.
    """
    sub = df.copy()
    target_items = items or sorted(sub["item_id"].unique())
    target_stores = stores or sorted(sub["store_id"].unique())

    records: List[Dict[str, Any]] = []

    for itm in target_items:
        for st in target_stores:
            series_df = sub[(sub["item_id"] == itm) & (sub["store_id"] == st)]
            if len(series_df) < 50:
                continue
            try:
                res = estimate_log_log_elasticity(series_df, item_id=itm, store_id=st)
                records.append({
                    "item_id": itm,
                    "store_id": st,
                    "elasticity": res["elasticity"],
                    "std_error": res["std_error"],
                    "ci_lower": res["ci_lower"],
                    "ci_upper": res["ci_upper"],
                    "p_value": res["p_value"],
                    "r2": res["r2"],
                    "n_obs": res["n_obs"],
                    "is_stat_sig": res["is_stat_sig_5pct"],
                })
            except Exception as e:
                warnings.warn(f"Failed to fit elasticity for {itm} in {st}: {e}", stacklevel=2)

    res_df = pd.DataFrame(records)
    if not res_df.empty:
        res_df = res_df.sort_values(by="elasticity").reset_index(drop=True)
    return res_df


def plot_elasticity_comparison(elasticity_df: pd.DataFrame) -> go.Figure:
    """Create an interactive forest plot comparing price elasticity point estimates and 95% CIs.

    Args:
        elasticity_df: DataFrame generated by compute_per_item_elasticities.

    Returns:
        Plotly Figure showing elasticity point estimates with error bars.
    """
    df_sorted = elasticity_df.sort_values(by="elasticity").copy()
    df_sorted["error_plus"] = df_sorted["ci_upper"] - df_sorted["elasticity"]
    df_sorted["error_minus"] = df_sorted["elasticity"] - df_sorted["ci_lower"]
    df_sorted["series_label"] = df_sorted["item_id"] + " (" + df_sorted["store_id"] + ")"

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df_sorted["elasticity"],
            y=df_sorted["series_label"],
            mode="markers",
            error_x=dict(
                type="data",
                symmetric=False,
                array=df_sorted["error_plus"],
                arrayminus=df_sorted["error_minus"],
                color="#1f77b4",
                thickness=1.5,
                width=5,
            ),
            marker=dict(size=8, color="#1f77b4"),
            hovertemplate=(
                "Series: %{y}<br>"
                "Elasticity: %{x:.2f}<br>"
                "<extra></extra>"
            ),
            name="Point Estimate (95% CI)",
        )
    )

    # Reference lines
    fig.add_vline(x=0.0, line_color="gray", line_dash="solid", line_width=1)
    fig.add_vline(x=-1.0, line_color="#d62728", line_dash="dash", line_width=1.5)

    fig.update_layout(
        title="Observational Price Elasticity across Items (with 95% Confidence Intervals)",
        xaxis_title="Price Elasticity (% Demand change per 1% Price change)",
        yaxis_title="Item & Store",
        template="plotly_white",
        height=max(500, len(df_sorted) * 22),
        annotations=[
            dict(
                x=-1.0,
                y=len(df_sorted) - 1,
                xref="x",
                yref="y",
                text="Unit Elastic (-1.0)",
                showarrow=True,
                arrowhead=2,
                ax=40,
                ay=-20,
            )
        ],
    )

    return fig
