# PricePilot: Progress Tracker

## Checklist

- [x] Phase 1: Data Pipeline & Preprocessing
- [x] Phase 2: Exploratory Data Analysis & Market Intelligence
- [x] Phase 3: Demand Modeling & Elasticity Estimation
- [x] Phase 4: Competitor Response Modeling
- [x] Phase 5: Stackelberg Game Optimization Engine
- [ ] Phase 6: Strategy Comparison & Evaluation Framework
- [ ] Phase 7: Sensitivity Analysis & Simulation Experiments
- [ ] Phase 8: Streamlit Dashboard Implementation
- [ ] Phase 9: Documentation, Packaging & Final Verification

## Decisions made

1. **Subset Selection**:
   - **Category**: `FOODS`, Department `FOODS_3`.
   - **Stores**: `CA_1` (Leader store) and `CA_2` (Follower/Competitor store), controlling for California SNAP schedule (`snap_CA`).
   - **Selected Items (15 items)**: `FOODS_3_090`, `FOODS_3_473`, `FOODS_3_723`, `FOODS_3_491`, `FOODS_3_734`, `FOODS_3_156`, `FOODS_3_202`, `FOODS_3_785`, `FOODS_3_324`, `FOODS_3_124`, `FOODS_3_227`, `FOODS_3_755`, `FOODS_3_351`, `FOODS_3_534`, `FOODS_3_704`.
   - **Primary Demonstration Item**: `FOODS_3_090` (high volume hero item, 8 distinct price levels from $1.00 to $1.60, CV = 0.0915, ~66.5 units/day in CA_1, ~29.4 in CA_2, low zero-rate).
2. **Missing Price Treatment**:
   - Rows where `sell_price` is missing correspond to weeks when an item was not yet on shelves or stocked. These rows are dropped rather than imputed.
3. **Leakage Prevention**:
   - Lag features (`lag_sales_1`, `7`, `14`, `28`) use grouped `.shift(k)`.
   - Rolling features (`rolling_mean_7`, `28`, `rolling_std_7`, `28`) apply `.shift(1)` before computing rolling statistics to ensure target day sales is strictly excluded.
4. **Chronological Splitting**:
   - Strict temporal ordering: Test set (last 28 days: 2016-03-28 to 2016-04-24, 840 rows), Validation set (28 days prior: 2016-02-29 to 2016-03-27, 840 rows), Train set (preceding history: 2011-01-29 to 2016-02-28, 55,710 rows).
5. **Phase 2 EDA Findings**:
   - `FOODS_3_090`, `FOODS_3_723`, and `FOODS_3_491` exhibit 8 distinct price levels with moderate price volatility ($CV \approx 0.08 - 0.10$).
   - Downward-sloping empirical demand identified for multiple products (`FOODS_3_124`, `FOODS_3_202`, `FOODS_3_704`, `FOODS_3_156`, `FOODS_3_351`, `FOODS_3_755`).
6. **Phase 3 Demand Modeling**:
   - **Champion Forecaster**: `Hist Gradient Boosting Regressor` (Test MAE: 3.482, RMSE: 5.915, WMAPE: 36.82%, $R^2$: 0.8081). Serialized to `models/demand_model.pkl`.
   - **Linear Demand Specification**: Extracted structural parameters $Q = a - b \cdot P + c \cdot P_F + d \cdot X$.
7. **Phase 4 Competitor Response Module**:
   - **Ground Truth Honesty**: M5 retail data contains NO external competitor prices. Modeled honestly in three labeled modes behind one unified common interface: `get_competitor_price(...)`.
   - Modes: `"rule"` (match, undercut, maintain, raise, adaptive), `"simulated_ml"` (Random Forest Classifier + Gradient Boosting Regressor), `"real"` (stub loader).
   - "Noisy ML" wrapper for sensitivity experiments with target accuracy levels (50% to 90%).
8. **Phase 5 Pricing Optimization Engine**:
   - **Linear Demand Engine**: Implemented $Q_L = a - b \cdot P_L + c \cdot P_F + d \cdot X$, clipped at $Q \ge 0$.
   - **Exogenous Cost Rule**: Profit is $\pi = (P - C) \cdot Q$. Cost $C$ is strictly an exogenous input parameter.
   - **Closed-Form Equivalence**: Analytical Stackelberg solution $P_L^* = \frac{C_L}{2} + \frac{a_{eff} + c \alpha}{2(b - c \beta)}$ verified to match numerical grid optimizer with zero discrepancy.
   - **Boundary & Clipping Edge Cases**: Optimizers evaluate across candidate price grid (default $[0.70 \times P_{base}, 1.30 \times P_{base}]$) and explicitly flag unreachable optima at grid boundaries (`at_min_boundary`, `at_max_boundary`, `boundary_warning`), negative margins, and demand choking ($Q=0$).

## Example Calls for Each Strategy

### Strategy A: Traditional Baseline (Ignoring Competitor)
```python
from src.optimization.baseline_optimizer import optimize_baseline

result_a = optimize_baseline(
    baseline_price=1.38,
    current_follower_price=1.37,
    cost=0.83,
    a=165.39,
    b=79.18,
    c=0.50,
)
print(f"Strategy A: Rec Price=${result_a.recommended_price:.2f}, Expected Profit=${result_a.expected_profit:.2f}")
# Output: Strategy A: Rec Price=$1.45, Expected Profit=$31.79
```

### Strategy B: Game-Theoretic Stackelberg (Rule-Based Follower)
```python
from src.optimization.game_optimizer import optimize_game_rule

result_b = optimize_game_rule(
    baseline_price=1.38,
    current_follower_price=1.37,
    cost=0.83,
    a=165.39,
    b=79.18,
    c=0.50,
    rule_params={"behavior": "match", "match_probability": 0.85},
)
print(f"Strategy B: Rec Price=${result_b.recommended_price:.2f}, Predicted Follower=${result_b.predicted_follower_price:.2f}, Expected Profit=${result_b.expected_profit:.2f}")
# Output: Strategy B: Rec Price=$1.48, Predicted Follower=$1.48, Expected Profit=$31.81
```

### Strategy C: ML-Aware Stackelberg (ML-Predicted Follower)
```python
from src.optimization.ml_optimizer import optimize_ml_stackelberg

result_c = optimize_ml_stackelberg(
    baseline_price=1.38,
    current_follower_price=1.37,
    cost=0.83,
    a=165.39,
    b=79.18,
    c=0.50,
    context={"recent_demand": 66.5, "sales_volatility": 2.5, "snap_CA": 1},
)
print(f"Strategy C: Rec Price=${result_c.recommended_price:.2f}, Predicted Follower=${result_c.predicted_follower_price:.2f}, Expected Profit=${result_c.expected_profit:.2f}")
# Output: Strategy C: Rec Price=$1.48, Predicted Follower=$1.49, Expected Profit=$31.82
```

## Strategy Comparison Summary (Hero Item `FOODS_3_090`)

| Metric | Strategy A (Baseline) | Strategy B (Game Rule: Match) | Strategy C (ML-Aware Stackelberg) |
| :--- | :---: | :---: | :---: |
| **Recommended Price ($)** | **$1.45** | **$1.48** | **$1.48** |
| **Predicted Follower Price ($)** | $1.37 (assumed fixed) | $1.48 (matched) | $1.49 (predicted) |
| **Expected Daily Demand (units)** | 51.1 | 49.0 | 49.0 |
| **Unit Margin ($)** | $0.62 | $0.65 | $0.65 |
| **Expected Profit ($)** | **$31.79** | **$31.81** | **$31.82** |
| **Profit vs Baseline** | Baseline (0.0%) | +0.06% | +0.09% |

## Interfaces (function signatures)

### `src.game.demand_function`
- `linear_demand(leader_price: float, follower_price: float, a: float, b: float, c: float = 0.50, context_shift: float = 0.0) -> float`
- `calculate_context_shift(context: Optional[Dict[str, Any]] = None, coefficients: Optional[Dict[str, float]] = None) -> float`

### `src.game.profit_function`
- `calculate_profit(price: float, cost: float, demand: float) -> float`
- `calculate_margin(price: float, cost: float) -> float`
- `calculate_margin_pct(price: float, cost: float) -> float`
- `summarize_financials(price: float, cost: float, demand: float) -> Dict[str, float]`

### `src.game.stackelberg`
- `closed_form_stackelberg(a: float, b: float, c: float, alpha: float, beta: float, cost: float, context_shift: float = 0.0) -> Dict[str, float]`
- `closed_form_baseline_monopoly(a: float, b: float, cost: float, c: float = 0.0, constant_follower_price: float = 0.0, context_shift: float = 0.0) -> Dict[str, float]`

### `src.optimization.baseline_optimizer`
- `class OptimizationResult(strategy: str, recommended_price: float, predicted_follower_price: float, expected_demand: float, expected_profit: float, expected_revenue: float, cost: float, unit_margin: float, margin_pct: float, baseline_price: float, grid_min: float, grid_max: float, at_min_boundary: bool, at_max_boundary: bool, boundary_warning: bool, q_clipped: bool, negative_margin: bool, grid_evaluations: pd.DataFrame)`
- `optimize_baseline(baseline_price: float, current_follower_price: float, cost: float, a: float, b: float, c: float = 0.50, context_shift: float = 0.0, min_factor: float = PRICE_GRID_MIN_FACTOR, max_factor: float = PRICE_GRID_MAX_FACTOR, num_steps: int = PRICE_GRID_NUM_STEPS, custom_grid: Optional[np.ndarray] = None) -> OptimizationResult`

### `src.optimization.game_optimizer`
- `optimize_game_rule(baseline_price: float, current_follower_price: float, cost: float, a: float, b: float, c: float = 0.50, context_shift: float = 0.0, rule_params: Optional[Dict[str, Any]] = None, min_factor: float = PRICE_GRID_MIN_FACTOR, max_factor: float = PRICE_GRID_MAX_FACTOR, num_steps: int = PRICE_GRID_NUM_STEPS, custom_grid: Optional[np.ndarray] = None) -> OptimizationResult`

### `src.optimization.ml_optimizer`
- `optimize_ml_stackelberg(baseline_price: float, current_follower_price: float, cost: float, a: float, b: float, c: float = 0.50, context: Optional[Dict[str, Any]] = None, context_shift: float = 0.0, competitor_model: Optional[CompetitorResponseModel] = None, target_accuracy: Optional[float] = None, min_factor: float = PRICE_GRID_MIN_FACTOR, max_factor: float = PRICE_GRID_MAX_FACTOR, num_steps: int = PRICE_GRID_NUM_STEPS, custom_grid: Optional[np.ndarray] = None) -> OptimizationResult`

## Known issues

None. Phases 1–5 passing 35/35 unit tests cleanly. Ready for Phase 6.
