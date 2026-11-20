# PricePilot

**Strategic Pricing Decision Support System**

PricePilot is a pricing decision support system built using Walmart M5 retail data, machine learning demand modeling, competitor response dynamics, and Stackelberg (leader-follower) game theory, complete with an interactive Streamlit dashboard.

## Project Structure

```
PricePilot/
├── app.py                 # Streamlit application entry point
├── requirements.txt       # Dependencies
├── README.md              # Project documentation
├── CONTEXT.md             # Project context and methodological guidelines
├── PROGRESS.md            # Progress tracking across phases
├── data/
│   ├── raw/m5/            # Raw M5 retail dataset
│   ├── processed/         # Cleaned and engineered features
│   └── sample/            # Small sample data for tests & quick runs
├── src/
│   ├── data/              # Data ingestion and preprocessing
│   ├── demand/            # Demand estimation and price elasticity models
│   ├── competitor/        # Competitor response modeling
│   ├── game/              # Stackelberg game theory engine
│   ├── optimization/      # Pricing optimization routines
│   ├── experiments/       # Strategy comparison and sensitivity benchmarks
│   └── utils/             # Config and helper utilities
├── dashboard/             # Dashboard UI components
├── models/                # Saved model artifacts
└── tests/                 # Unit and integration test suite
```

## Quick Start

```bash
# Navigate to project directory
cd PricePilot

# Install dependencies
pip install -r requirements.txt

# Run the test suite (47 tests across all modules)
pytest tests/

# Launch the interactive Streamlit dashboard
streamlit run app.py
```

## Interactive Dashboard Pages

1. **Executive Summary (`dashboard/overview.py`)**: High-level KPIs, optimal recommended price vs baseline, expected profit and demand, and true-world simulation benchmark summary.
2. **Market Intelligence (`dashboard/market.py`)**: Product and store selectors across 15 continuous M5 catalog items, time series trends, empirical price-quantity demand curves, seasonality breakdowns, and promotion/discount effects.
3. **Demand Forecasting (`dashboard/demand.py`)**: Out-of-sample benchmark comparison (Hist Gradient Boosting champion), actual vs predicted time series plots, econometric price elasticity estimates with observational caveats, and interactive price-response curve with extrapolation boundaries.
4. **Competitor Intelligence (`dashboard/competitor.py`)**: Common interface supporting 3 labeled modes (`simulated_ml`, `rule`, `real`) with prominent methodological honesty notices, response probability estimation, and reaction curves.
5. **Price Optimizer (`dashboard/optimizer.py`)**: Real-time Stackelberg optimization comparing Strategy A (Baseline), Strategy B (Rule-based Game), and Strategy C (ML-Aware Game) over a configurable candidate price grid, with boundary and margin edge case alerts.
6. **Strategy Comparison (`dashboard/comparison.py`)**: 40-run multi-period simulation benchmarks under Common Random Numbers (CRN), featuring cumulative profit growth, downside risk (5th percentile), profit variance distributions, and honest win/loss/tie breakdowns.
7. **Sensitivity Analysis (`dashboard/sensitivity.py`)**: Precomputed parameter sweeps covering response strength $\times$ ML accuracy heatmaps, asymmetric error regimes (optimistic vs pessimistic), break-even accuracy thresholds, competitor behavior policies, and cost/volatility robustness.

