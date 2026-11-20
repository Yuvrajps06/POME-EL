"""Exploratory Data Analysis (EDA) and Market Intelligence module.

Provides reusable Plotly visualization functions and KPI summaries for
pricing dynamics, demand patterns, seasonality, and elasticity readiness.
"""

from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def compute_market_kpis(
    df: pd.DataFrame,
    item_id: Optional[str] = None,
    store_id: Optional[str] = None,
) -> pd.DataFrame:
    """Compute market intelligence KPIs for items and stores.

    Metrics include:
    - Average price
    - Average daily demand
    - Demand trend slope direction (Increasing, Decreasing, Stable)
    - Price volatility (Coefficient of Variation) with categorical label
    - Zero-sales share
    - Distinct price count

    Args:
        df: Processed DataFrame containing sales, sell_price, item_id, store_id.
        item_id: Optional specific item to filter.
        store_id: Optional specific store to filter.

    Returns:
        pd.DataFrame of KPIs per (item_id, store_id).
    """
    sub = df.copy()
    if item_id:
        sub = sub[sub["item_id"] == item_id]
    if store_id:
        sub = sub[sub["store_id"] == store_id]

    records: List[Dict[str, Union[str, float, int, bool]]] = []

    groups = sub.groupby(["item_id", "store_id"], observed=True)

    for (itm, st), group in groups:
        if len(group) == 0:
            continue

        prices = group["sell_price"].dropna()
        sales = group["sales"].dropna()

        avg_price = float(prices.mean())
        price_std = float(prices.std()) if len(prices) > 1 else 0.0
        cv = (price_std / avg_price) if avg_price > 0 else 0.0

        if cv < 0.05:
            volatility_label = "Low"
        elif cv <= 0.15:
            volatility_label = "Moderate"
        else:
            volatility_label = "High"

        avg_demand = float(sales.mean())
        zero_share = float((sales == 0).mean())
        n_distinct = int(prices.nunique())

        # Demand trend: slope from OLS on day sequence
        if len(sales) > 1:
            x = np.arange(len(sales))
            slope, _ = np.polyfit(x, sales.values, 1)
            # Define threshold based on daily demand scale
            norm_slope = slope * len(sales) / max(avg_demand, 1.0)
            if norm_slope > 0.15:
                trend_label = "Increasing"
            elif norm_slope < -0.15:
                trend_label = "Decreasing"
            else:
                trend_label = "Stable"
        else:
            slope = 0.0
            trend_label = "Stable"

        # Suitability for Stackelberg elasticity modeling:
        # Requires >= 5 distinct prices, low zero share (< 30%), moderate/high volatility
        is_recommended = (
            n_distinct >= 5
            and zero_share < 0.30
            and volatility_label in ["Moderate", "High"]
        )

        records.append({
            "item_id": itm,
            "store_id": st,
            "avg_price": round(avg_price, 2),
            "price_min": round(float(prices.min()), 2),
            "price_max": round(float(prices.max()), 2),
            "price_cv": round(cv, 4),
            "price_volatility": volatility_label,
            "distinct_prices": n_distinct,
            "avg_daily_demand": round(avg_demand, 2),
            "zero_sales_share": round(zero_share, 3),
            "demand_slope": round(float(slope), 5),
            "demand_trend": trend_label,
            "recommended_for_modeling": is_recommended,
        })

    return pd.DataFrame(records)


def plot_sales_and_price_over_time(
    df: pd.DataFrame,
    item_id: str,
    store_id: str,
) -> go.Figure:
    """Plot dual-axis time-series of daily sales and weekly sell prices.

    Args:
        df: DataFrame with date, sales, sell_price, item_id, store_id.
        item_id: Target item ID.
        store_id: Target store ID.

    Returns:
        Plotly Figure with sales on primary y-axis and price on secondary y-axis.
    """
    sub = df[(df["item_id"] == item_id) & (df["store_id"] == store_id)].copy()
    sub = sub.sort_values(by="date")

    fig = make_subplots(
        specs=[[{"secondary_y": True}]],
        subplot_titles=[f"Demand & Price Dynamics: {item_id} ({store_id})"],
    )

    # Sales trace (daily volume)
    fig.add_trace(
        go.Bar(
            x=sub["date"],
            y=sub["sales"],
            name="Daily Sales",
            marker_color="rgba(31, 119, 180, 0.4)",
            hovertemplate="Date: %{x|%Y-%m-%d}<br>Sales: %{y}<extra></extra>",
        ),
        secondary_y=False,
    )

    # Price trace (stepped price line)
    fig.add_trace(
        go.Scatter(
            x=sub["date"],
            y=sub["sell_price"],
            name="Selling Price ($)",
            line=dict(color="#d62728", width=2.5, shape="hv"),
            hovertemplate="Date: %{x|%Y-%m-%d}<br>Price: $%{y:.2f}<extra></extra>",
        ),
        secondary_y=True,
    )

    fig.update_layout(
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(title="Date"),
    )
    fig.update_yaxes(title_text="Units Sold", secondary_y=False)
    fig.update_yaxes(title_text="Sell Price ($)", secondary_y=True)

    return fig


def plot_price_vs_quantity(
    df: pd.DataFrame,
    item_id: str,
    store_id: str,
    log_scale: bool = False,
    add_trendline: bool = True,
) -> go.Figure:
    """Plot price vs quantity scatter plot with optional log-log transform.

    Args:
        df: Processed DataFrame.
        item_id: Target item ID.
        store_id: Target store ID.
        log_scale: Whether to plot log(price) vs log(sales+1).
        add_trendline: Whether to fit an OLS regression trendline.

    Returns:
        Plotly Figure illustrating demand curve shape and empirical elasticity.
    """
    sub = df[(df["item_id"] == item_id) & (df["store_id"] == store_id)].copy()

    if log_scale:
        x_col = sub["log_price"] if "log_price" in sub.columns else np.log(sub["sell_price"])
        y_col = sub["log_sales"] if "log_sales" in sub.columns else np.log1p(sub["sales"])
        x_label = "ln(Price)"
        y_label = "ln(Sales + 1)"
        title_suffix = "(Log-Log Elasticity Space)"
    else:
        x_col = sub["sell_price"]
        y_col = sub["sales"]
        x_label = "Selling Price ($)"
        y_label = "Daily Demand (Units)"
        title_suffix = "(Natural Units)"

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x_col,
            y=y_col,
            mode="markers",
            marker=dict(
                size=6,
                color="#1f77b4",
                opacity=0.45,
                line=dict(width=0.5, color="#0b3c61"),
            ),
            name="Daily Observations",
            hovertemplate=f"{x_label}: %{{x:.2f}}<br>{y_label}: %{{y:.2f}}<extra></extra>",
        )
    )

    if add_trendline and len(sub) > 2:
        valid_mask = np.isfinite(x_col) & np.isfinite(y_col)
        if valid_mask.sum() > 2:
            poly = np.polyfit(x_col[valid_mask], y_col[valid_mask], 1)
            x_line = np.linspace(x_col[valid_mask].min(), x_col[valid_mask].max(), 100)
            y_line = np.polyval(poly, x_line)

            slope_label = f"OLS Slope: {poly[0]:.2f}"
            fig.add_trace(
                go.Scatter(
                    x=x_line,
                    y=y_line,
                    mode="lines",
                    line=dict(color="#d62728", width=2.5, dash="dash"),
                    name=f"Demand Fit ({slope_label})",
                )
            )

    fig.update_layout(
        title=f"Empirical Demand Curve: {item_id} [{store_id}] {title_suffix}",
        xaxis_title=x_label,
        yaxis_title=y_label,
        template="plotly_white",
    )
    return fig


def plot_sales_distribution(
    df: pd.DataFrame,
    item_id: Optional[str] = None,
    store_id: Optional[str] = None,
) -> go.Figure:
    """Plot sales distribution histogram and box plot.

    Args:
        df: Processed DataFrame.
        item_id: Optional target item.
        store_id: Optional target store.

    Returns:
        Plotly Figure showing demand distribution and zero-sales density.
    """
    sub = df.copy()
    if item_id:
        sub = sub[sub["item_id"] == item_id]
    if store_id:
        sub = sub[sub["store_id"] == store_id]

    fig = px.histogram(
        sub,
        x="sales",
        nbins=40,
        marginal="box",
        color_discrete_sequence=["#2ca02c"],
        title=f"Daily Demand Distribution {f'for {item_id}' if item_id else 'All Items'}",
        labels={"sales": "Units Sold"},
        template="plotly_white",
    )
    fig.update_layout(bargap=0.08)
    return fig


def plot_seasonality(
    df: pd.DataFrame,
    item_id: Optional[str] = None,
    store_id: Optional[str] = None,
    freq: str = "dow",
) -> go.Figure:
    """Plot demand seasonality across day-of-week, month, or event flags.

    Args:
        df: Processed DataFrame.
        item_id: Optional item filter.
        store_id: Optional store filter.
        freq: Seasonality dimension: 'dow' (day of week), 'month', or 'event'.

    Returns:
        Plotly Figure visualizing demand variation across the seasonal factor.
    """
    sub = df.copy()
    if item_id:
        sub = sub[sub["item_id"] == item_id]
    if store_id:
        sub = sub[sub["store_id"] == store_id]

    if freq == "dow":
        dow_order = ["Saturday", "Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
        grouped = (
            sub.groupby("weekday", observed=True)["sales"]
            .agg(mean="mean", std="std", count="count")
            .reindex(dow_order)
            .reset_index()
        )
        fig = px.bar(
            grouped,
            x="weekday",
            y="mean",
            error_y=grouped["std"] / np.sqrt(grouped["count"]),
            title=f"Seasonality: Day of Week Average Demand {f'({item_id})' if item_id else ''}",
            labels={"weekday": "Day of Week", "mean": "Average Units Sold"},
            template="plotly_white",
            color_discrete_sequence=["#1f77b4"],
        )

    elif freq == "month":
        grouped = (
            sub.groupby("month", observed=True)["sales"]
            .agg(mean="mean", std="std", count="count")
            .reset_index()
        )
        month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        grouped["month_name"] = grouped["month"].apply(lambda m: month_names[int(m) - 1])

        fig = px.bar(
            grouped,
            x="month_name",
            y="mean",
            error_y=grouped["std"] / np.sqrt(grouped["count"]),
            title=f"Seasonality: Monthly Demand Profile {f'({item_id})' if item_id else ''}",
            labels={"month_name": "Month", "mean": "Average Units Sold"},
            template="plotly_white",
            color_discrete_sequence=["#ff7f0e"],
        )

    elif freq == "event":
        sub["event_status"] = np.where(sub["has_event"] == 1, "Holiday / Event", "Standard Day")
        grouped = (
            sub.groupby("event_status", observed=True)["sales"]
            .agg(mean="mean", std="std", count="count")
            .reset_index()
        )
        fig = px.bar(
            grouped,
            x="event_status",
            y="mean",
            error_y=grouped["std"] / np.sqrt(grouped["count"]),
            title=f"Event vs Non-Event Impact {f'({item_id})' if item_id else ''}",
            labels={"event_status": "Calendar Status", "mean": "Average Units Sold"},
            template="plotly_white",
            color="event_status",
            color_discrete_map={"Holiday / Event": "#d62728", "Standard Day": "#2ca02c"},
        )
    else:
        raise ValueError(f"Unknown freq: {freq}. Supported: 'dow', 'month', 'event'.")

    return fig


def plot_price_change_vs_sales_change(
    df: pd.DataFrame,
    item_id: Optional[str] = None,
    store_id: Optional[str] = None,
) -> go.Figure:
    """Plot price change percentage vs subsequent sales change percentage.

    Calculates weekly percentage changes to filter daily stockout noise and isolate
    the immediate reaction to price revisions.

    Args:
        df: Processed DataFrame.
        item_id: Optional item filter.
        store_id: Optional store filter.

    Returns:
        Plotly Figure showing elasticity scatter of price adjustments.
    """
    sub = df.copy()
    if item_id:
        sub = sub[sub["item_id"] == item_id]
    if store_id:
        sub = sub[sub["store_id"] == store_id]

    # Aggregate by week to mirror Walmart pricing decisions
    weekly = (
        sub.groupby(["item_id", "store_id", "wm_yr_wk"], observed=True)
        .agg({"sell_price": "mean", "sales": "sum"})
        .reset_index()
    )

    weekly["pct_price_change"] = (
        weekly.groupby(["item_id", "store_id"])["sell_price"].pct_change() * 100.0
    )
    weekly["pct_sales_change"] = (
        weekly.groupby(["item_id", "store_id"])["sales"].pct_change() * 100.0
    )

    # Filter out zero-price changes to focus on actual repricing moments
    active_changes = weekly[weekly["pct_price_change"].abs() > 0.1].dropna(
        subset=["pct_price_change", "pct_sales_change"]
    )

    # Clip extreme percentage outliers for visual clarity
    clipped = active_changes.copy()
    clipped["pct_sales_change"] = clipped["pct_sales_change"].clip(-150, 250)

    fig = px.scatter(
        clipped,
        x="pct_price_change",
        y="pct_sales_change",
        color="item_id",
        trendline="ols",
        title=f"Price Revisions vs Demand Response (% Change) {f'({item_id})' if item_id else ''}",
        labels={
            "pct_price_change": "Price Change (%)",
            "pct_sales_change": "Demand Change (%)",
        },
        template="plotly_white",
    )
    fig.add_hline(y=0, line_dash="dot", line_color="gray")
    fig.add_vline(x=0, line_dash="dot", line_color="gray")

    return fig


def plot_promotion_effect(
    df: pd.DataFrame,
    item_id: Optional[str] = None,
    store_id: Optional[str] = None,
) -> go.Figure:
    """Compare demand when price is discounted below average vs regular price.

    Args:
        df: Processed DataFrame with 'is_discounted' and 'sales'.
        item_id: Optional item filter.
        store_id: Optional store filter.

    Returns:
        Plotly Figure comparing promotional vs non-promotional demand.
    """
    sub = df.copy()
    if item_id:
        sub = sub[sub["item_id"] == item_id]
    if store_id:
        sub = sub[sub["store_id"] == store_id]

    sub["promo_label"] = np.where(
        sub["is_discounted"] == 1,
        "Discounted (Below Historical Avg)",
        "Regular / Above Avg",
    )

    fig = px.box(
        sub,
        x="promo_label",
        y="sales",
        color="promo_label",
        points="outliers",
        title=f"Promotional Uplift: Demand under Discounted vs Regular Pricing {f'({item_id})' if item_id else ''}",
        labels={"promo_label": "Price Regime", "sales": "Daily Units Sold"},
        color_discrete_map={
            "Discounted (Below Historical Avg)": "#2ca02c",
            "Regular / Above Avg": "#1f77b4",
        },
        template="plotly_white",
    )
    return fig


def plot_product_comparison(
    df: pd.DataFrame,
    item_ids: Optional[List[str]] = None,
    metric: str = "sales",
) -> go.Figure:
    """Compare candidate products across sales volume or average price.

    Args:
        df: Processed DataFrame.
        item_ids: List of item IDs to include.
        metric: 'sales' (average daily sales) or 'price' (average price).

    Returns:
        Plotly Figure comparing products across stores.
    """
    sub = df.copy()
    if item_ids:
        sub = sub[sub["item_id"].isin(item_ids)]

    if metric == "sales":
        agg = (
            sub.groupby(["item_id", "store_id"], observed=True)["sales"]
            .mean()
            .reset_index()
        )
        title = "Product Comparison: Average Daily Sales by Store"
        y_label = "Average Daily Sales (Units)"
        y_col = "sales"
    else:
        agg = (
            sub.groupby(["item_id", "store_id"], observed=True)["sell_price"]
            .mean()
            .reset_index()
        )
        title = "Product Comparison: Average Selling Price by Store"
        y_label = "Average Price ($)"
        y_col = "sell_price"

    fig = px.bar(
        agg,
        x="item_id",
        y=y_col,
        color="store_id",
        barmode="group",
        title=title,
        labels={"item_id": "Item ID", y_col: y_label, "store_id": "Store"},
        template="plotly_white",
        color_discrete_sequence=["#1f77b4", "#ff7f0e"],
    )
    fig.update_xaxes(tickangle=45)
    return fig
