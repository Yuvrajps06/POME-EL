# PricePilot: Strategic Pricing Decision Support System

**PricePilot** is a research-grade strategic pricing decision support system built using authentic Walmart M5 retail sales data, machine learning demand modeling, a three-mode competitor response engine, and Stackelberg (leader-follower) game theory, presented through an interactive multi-page Streamlit dashboard.

---

## Research Question

> **Does incorporating machine-learning-predicted competitor reactions into a Stackelberg pricing decision improve realized firm profit over traditional non-game baseline pricing, and under what market conditions?**

---

## Architecture Diagram

```
+---------------------------------------------------------------------------------------------------+
|                                      PRICEPILOT ARCHITECTURE                                      |
+---------------------------------------------------------------------------------------------------+

 [Raw Walmart M5 Data]
  - sales_train_validation.csv
  - calendar.csv
  - sell_prices.csv
          |
          v
 [Phase 1: Ingestion & Feature Engineering]
  - Filter: FOODS_3 catalog, Stores CA_1 (Leader) & CA_2 (Competitor)
  - Chronological Split: Train (55,710) < Val (840) < Test (840)
  - Leakage-Free Features: Lags (1, 7, 14, 28), Rolling Means/Stds (Pre-shifted)
          |
          +-----------------------------------+
          |                                   |
          v                                   v
 [Phase 2 & 3: Demand Modeling]      [Phase 4: Competitor Response Module]
  - Champion: HistGradientBoosting    - Mode 'rule': match, undercut, maintain, raise
  - Log-log Econometric Elasticity    - Mode 'simulated_ml': RF Classifier + GB Regressor
  - Linear Specification:             - Mode 'real': Schema-verified external data stub
    Q_L = a - b*P_L + c*P_F + d*X     - Noisy wrapper: Controlled accuracy degradation
          |                                   |
          +-----------------+-----------------+
                            |
                            v
              [Phase 5: Stackelberg Pricing Engine]
               Max pi_L = (P_L - C_L) * Q_L(P_L, R(P_L, X))
               Bounded Price Search Grid: [0.70*P0, 1.30*P0]
               Analytical Closed-Form Solver Equivalence
                            |
                            +-----------------------------------+
                            |                                   |
                            v                                   v
              [Phase 6: Strategy Comparison]      [Phase 7: Sensitivity Sweeps]
               - MarketEnvironment Simulator       - Response Strength x ML Accuracy
               - 40 Runs, 28-day Horizon (CRN)     - Symmetric vs Asymmetric Errors
               - Strategies A, B, and C            - Behavior Policies & Delays (0, 1, 3)
                            |                                   |
                            +-----------------+-----------------+
                                              |
                                              v
                              [Phase 8: Streamlit Dashboard]
                               - Executive Summary
                               - Market Intelligence
                               - Demand Forecasting
                               - Competitor Response
                               - Real-Time Price Optimizer
                               - Strategy Benchmarking
                               - Sensitivity Sweeps
```

---

## Setup & Installation

### Requirements
- Python 3.10+ (tested on Python 3.14)
- Packages: `pandas`, `numpy`, `scikit-learn`, `scipy`, `statsmodels`, `plotly`, `streamlit`, `joblib`, `pytest`, `fastparquet`, `pyarrow`

### Installation
```bash
# Clone or navigate to the project directory
cd PricePilot

# Install dependencies
pip install -r requirements.txt
```

---

## Data Download Instructions

The Walmart M5 competition dataset files are stored in `data/raw/m5/`:
- `sales_train_validation.csv`
- `calendar.csv`
- `sell_prices.csv`

If raw files are not present, download the authentic dataset using the provided curl script or from the official Kaggle M5 competition:
```bash
# Automatic extraction from mirror:
mkdir -p data/raw/m5
curl -L -o data/raw/m5/m5.zip https://github.com/Nixtla/m5-forecasts/raw/main/datasets/m5.zip
unzip -q data/raw/m5/m5.zip -d data/raw/m5/
rm data/raw/m5/m5.zip
```

---

## How to Run

### 1. Run the Full Pipeline End-to-End
Runs data ingestion, feature engineering, model training, simulation benchmarking, and sensitivity caching:
```bash
make pipeline
# OR
python3 scripts/run_pipeline.py
```

### 2. Run the Full Test Suite
Executes 51 unit tests covering all data, modeling, game-theoretic, simulation, and dashboard integration routines:
```bash
make test
# OR
pytest tests/
```

### 3. Launch the Interactive Streamlit Dashboard
Launches the 7-page interactive decision support application:
```bash
make dashboard
# OR
streamlit run app.py
```
Open `http://localhost:8501` in your browser.

---

## Dashboard Pages Guide

1. **Executive Summary (`dashboard/overview.py`)**: Key decision metrics, current vs recommended pricing, expected daily profits, and multi-period simulation benchmarks.
2. **Market Intelligence (`dashboard/market.py`)**: Catalog and store selectors, price/demand time series, empirical price-quantity relationships, multi-period seasonality, and discount effects.
3. **Demand Forecasting (`dashboard/demand.py`)**: Out-of-sample benchmark comparison (HistGradientBoosting champion), actual vs predicted demand, log-log elasticity estimates with observational caveats, and extrapolation boundaries.
4. **Competitor Intelligence (`dashboard/competitor.py`)**: Mode selector (`simulated_ml`, `rule`, `real`), persistent methodological honesty notices, response probability curves, and schema validation.
5. **Price Optimizer (`dashboard/optimizer.py`)**: Real-time pricing optimization evaluating Strategy A (Baseline), Strategy B (Game Rule), and Strategy C (ML-Aware Stackelberg) across candidate price grids with boundary edge case alerts.
6. **Strategy Comparison (`dashboard/comparison.py`)**: 40-run ground-truth simulation benchmarking under Common Random Numbers (CRN), downside risk (5th percentile), and win/loss/tie analysis.
7. **Sensitivity Analysis (`dashboard/sensitivity.py`)**: Precomputed parameter sweeps covering response strength $\times$ accuracy heatmaps, break-even accuracy thresholds, asymmetric error impacts, competitor delays, and parameter uncertainty.

---

## Methodological Rules & Limitations

PricePilot adheres strictly to the non-negotiable methodological principles defined in `CONTEXT.md`:

1. **No External Competitor Data in M5**:
   - The Walmart M5 retail dataset contains store-level prices for Walmart locations but **no external retail competitor prices**.
   - Competitor responses are honestly evaluated in three labeled modes: `"simulated_ml"`, `"rule"`, and `"real"` (stub loader). All outputs state clearly: *Simulated results reflect recovery of behavioral reaction rules, not real-world competitor accuracy.*
2. **Exogenous Unit Cost**:
   - Walmart M5 contains no wholesale cost data. Unit cost $C$ is strictly an external scenario input parameter and is **never inferred from selling prices**.
3. **Strict Chronological Splitting**:
   - All dataset partitioning enforces temporal order: Train ($t_1 \dots t_{val-1}$) < Validation (28 days) < Test (final 28 days). No future leakage.
4. **Observational Elasticity Caveat**:
   - Price elasticities derived from observational regression reflect historical equilibrium co-movements and are **correlational, not causal**.
5. **Bounded Candidate Grids**:
   - Price optimization is strictly constrained to a reasonable candidate range around historical observed prices ($[0.70 \times P_{base}, 1.30 \times P_{base}]$) to prevent catastrophic model extrapolation.
6. **Decision-Level Evaluation (Beyond Accuracy)**:
   - Models are judged not merely on $R^2$ or classification accuracy, but on realized profit in a ground-truth stochastic market environment.
