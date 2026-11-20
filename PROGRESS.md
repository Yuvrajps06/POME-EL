# PricePilot: Progress Tracker

## Checklist

- [x] Phase 1: Data Pipeline & Preprocessing
- [x] Phase 2: Exploratory Data Analysis & Market Intelligence
- [x] Phase 3: Demand Modeling & Elasticity Estimation
- [ ] Phase 4: Competitor Response Modeling
- [ ] Phase 5: Stackelberg Game Optimization Engine
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
6. **Phase 3 Modeling Decisions**:
   - **Champion Forecaster**: `Hist Gradient Boosting Regressor` achieved the lowest test MAE (3.482), lowest test RMSE (5.915), lowest test WMAPE (36.82%), and highest test $R^2$ (0.8081), beating the seasonal naive benchmark by 21.6% MAE reduction. Serialized to `models/demand_model.pkl`.
   - **Extrapolation Protection**: `DemandModel.price_response_curve` issues explicit user warnings when pricing decisions probe outside historical observed bounds [$0.94, $5.98].
   - **Linear Demand Specification**: Extracted structural parameters $Q = a - b \cdot P + c \cdot P_F + d \cdot X$ for the game theory engine. For items with positive unconstrained OLS slopes, regularized $b > 0$ based on unit elasticity was structurally imposed to ensure valid downward-sloping Stackelberg reaction functions. Cross-price elasticity $c$ initialized to $0.50$ (to be calibrated in Phase 4).

## Model Comparison (Chronological Validation & Test)

| Model | Val MAE | Val RMSE | Val WMAPE | Val R² | Test MAE | Test RMSE | Test WMAPE | Test R² |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Hist Gradient Boosting** (Champion) | **3.111** | **5.920** | **35.28%** | **0.8139** | **3.482** | **5.915** | **36.82%** | **0.8081** |
| **Ridge Regression** | 3.306 | 6.182 | 37.48% | 0.7971 | 3.618 | 6.136 | 38.25% | 0.7935 |
| **Linear Regression** | 3.305 | 6.184 | 37.47% | 0.7969 | 3.623 | 6.140 | 38.30% | 0.7932 |
| **Random Forest** | 3.570 | 6.314 | 40.48% | 0.7883 | 3.823 | 6.281 | 40.42% | 0.7836 |
| **Seasonal Naive** (Lag-7 Baseline) | 3.908 | 7.803 | 44.32% | 0.6767 | 4.443 | 7.743 | 46.97% | 0.6712 |

## Observational Elasticity & Caveats

> [!CAUTION]
> **Methodological Caveat**: Price elasticity estimated from historical regression is observational, NOT causal. It captures historical co-movements between price and sales volume conditional on observed calendar, holiday, and SNAP controls. It does not represent a randomized counterfactual experiment and may remain confounded by unobserved demand shocks, promotional displays, or store traffic.

### Top Elasticity Estimates (Statsmodels OLS with Controls)

| Item ID | Store | Elasticity ($\hat{\beta}_1$) | Std Error | 95% Confidence Interval | p-value | R² |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `FOODS_3_124` | `CA_1` | **-5.223** | 0.440 | [-6.087, -4.360] | < 0.0001 | 0.1718 |
| `FOODS_3_124` | `CA_2` | **-4.950** | 0.409 | [-5.752, -4.148] | < 0.0001 | 0.1956 |
| `FOODS_3_351` | `CA_1` | **-4.949** | 0.308 | [-5.552, -4.345] | < 0.0001 | 0.2416 |
| `FOODS_3_755` | `CA_1` | **-4.758** | 0.332 | [-5.409, -4.107] | < 0.0001 | 0.2178 |
| `FOODS_3_202` | `CA_1` | **-4.668** | 0.374 | [-5.402, -3.934] | < 0.0001 | 0.2058 |
| `FOODS_3_704` | `CA_1` | **-4.614** | 0.275 | [-5.153, -4.074] | < 0.0001 | 0.2226 |
| `FOODS_3_156` | `CA_1` | **-4.019** | 0.388 | [-4.780, -3.259] | < 0.0001 | 0.0928 |
| `FOODS_3_704` | `CA_2` | **-3.797** | 0.264 | [-4.314, -3.280] | < 0.0001 | 0.1703 |
| `FOODS_3_755` | `CA_2` | **-3.131** | 0.303 | [-3.725, -2.537] | < 0.0001 | 0.1153 |
| `FOODS_3_351` | `CA_2` | **-2.874** | 0.294 | [-3.451, -2.297] | < 0.0001 | 0.1072 |
| `FOODS_3_202` | `CA_2` | **-2.635** | 0.342 | [-3.306, -1.963] | < 0.0001 | 0.1574 |

### Game Engine Linear Demand Parameters ($Q = a - b \cdot P + c \cdot P_F + d \cdot X$)

| Item ID | Store | Intercept $a$ | Own-Price Slope $b$ | Cross-Price Slope $c$ | R² | Mean Price ($) | Mean Sales |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `FOODS_3_090` | `CA_1` | 165.394 | 79.181 | 0.50 (cfg) | 0.1948 | $1.38 | 66.49 |
| `FOODS_3_090` | `CA_2` | 58.724 | 21.402 | 0.50 (cfg) | 0.1572 | $1.37 | 29.36 |
| `FOODS_3_124` | `CA_1` | 32.340 | 5.080 | 0.50 (cfg) | 0.2419 | $5.38 | 5.62 |
| `FOODS_3_124` | `CA_2` | 21.532 | 3.562 | 0.50 (cfg) | 0.2087 | $5.38 | 3.21 |
| `FOODS_3_202` | `CA_1` | 62.783 | 10.717 | 0.50 (cfg) | 0.2450 | $4.32 | 16.40 |
| `FOODS_3_202` | `CA_2` | 12.213 | 1.512 | 0.50 (cfg) | 0.1574 | $4.31 | 8.54 |
| `FOODS_3_704` | `CA_1` | 27.766 | 5.241 | 0.50 (cfg) | 0.2280 | $4.31 | 4.61 |
| `FOODS_3_704` | `CA_2` | 16.433 | 3.204 | 0.50 (cfg) | 0.1727 | $4.31 | 2.94 |
| `FOODS_3_156` | `CA_1` | 31.141 | 6.891 | 0.50 (cfg) | 0.0939 | $3.85 | 5.81 |

## Interfaces (function signatures)

### `src.demand.demand_model`
- `class DemandModel(model: Any, feature_names: List[str], min_observed_price: float, max_observed_price: float, model_name: str)`
  - `predict(X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray`
  - `predict_demand(price: float, context_features: Union[Dict[str, Any], pd.Series, pd.DataFrame]) -> float`
  - `price_response_curve(context_features: Union[Dict[str, Any], pd.Series], price_grid: Optional[np.ndarray] = None, min_price: Optional[float] = None, max_price: Optional[float] = None, n_points: int = 30) -> pd.DataFrame`
- `fit_linear_approximation(df: pd.DataFrame, item_id: str, store_id: str, default_c: float = 0.50) -> Dict[str, Any]`
- `save_demand_model(wrapper: DemandModel, path: Optional[Union[str, Path]] = None) -> Path`
- `load_demand_model(path: Optional[Union[str, Path]] = None) -> DemandModel`

### `src.demand.elasticity`
- `estimate_log_log_elasticity(df: pd.DataFrame, item_id: Optional[str] = None, store_id: Optional[str] = None, controls: Optional[List[str]] = None) -> Dict[str, Any]`
- `compute_per_item_elasticities(df: pd.DataFrame, items: Optional[List[str]] = None, stores: Optional[List[str]] = None) -> pd.DataFrame`
- `plot_elasticity_comparison(elasticity_df: pd.DataFrame) -> go.Figure`

### `src.demand.forecasting`
- `class SeasonalNaiveForecaster(lag_col: str = "lag_sales_7")`
  - `fit(X: pd.DataFrame, y: Optional[pd.Series] = None) -> SeasonalNaiveForecaster`
  - `predict(X: pd.DataFrame) -> np.ndarray`
- `safe_wmape(y_true: Union[pd.Series, np.ndarray], y_pred: Union[pd.Series, np.ndarray]) -> float`
- `train_and_evaluate_models(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: Optional[List[str]] = None, target_col: str = "sales", save_best: bool = True) -> Tuple[Dict[str, Any], pd.DataFrame, str]`
- `plot_actual_vs_predicted(actual: Union[pd.Series, np.ndarray], predicted: Union[pd.Series, np.ndarray], dates: Union[pd.Series, np.ndarray], title: str = "Actual vs Predicted Demand") -> go.Figure`

## Known issues

None. Phases 1, 2, and 3 passing 20/20 unit tests cleanly. Ready for Phase 4.
