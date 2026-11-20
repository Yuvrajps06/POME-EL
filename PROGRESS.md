# PricePilot: Progress Tracker

## Checklist

- [x] Phase 1: Data Pipeline & Preprocessing
- [x] Phase 2: Exploratory Data Analysis & Market Intelligence
- [x] Phase 3: Demand Modeling & Elasticity Estimation
- [x] Phase 4: Competitor Response Modeling
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
6. **Phase 3 Demand Modeling**:
   - **Champion Forecaster**: `Hist Gradient Boosting Regressor` (Test MAE: 3.482, RMSE: 5.915, WMAPE: 36.82%, $R^2$: 0.8081). Serialized to `models/demand_model.pkl`.
   - **Linear Demand Specification**: Extracted structural parameters $Q = a - b \cdot P + c \cdot P_F + d \cdot X$.
7. **Phase 4 Competitor Response Module**:
   - **Ground Truth Honesty**: M5 retail data contains NO external competitor prices. Modeled honestly in three labeled modes behind one unified common interface.
   - **Common Interface**: `get_competitor_price(leader_price, current_follower_price, context, mode, params) -> (follower_price, probability_of_increase)`
   - **Modes Implemented**:
     1. `"rule"`: Configurable rule-based response (behaviors: `match`, `undercut`, `maintain`, `raise`, `adaptive`; parameters: `match_probability`, `response_strength`, `undercut_margin`, `noise_std`).
     2. `"simulated_ml"`: Trained on 5,000 synthetic interaction scenarios generated from known behavioral rules with noise. Champion classifier (`RandomForestClassifier`, 96.1% test accuracy) predicts price increase probability; regressor (`HistGradientBoostingRegressor`) predicts reaction magnitude. Serialized to `models/competitor_model.pkl`. All outputs labeled: *"SIMULATED DATA (not real-world accuracy)"*.
     3. `"real"`: Stub loader enforcing documented schema `['Date', 'Our_Price', 'Competitor_Price', 'Our_Sales', 'Competitor_Sales']`. Raises clear `FileNotFoundError` if unsupplied.
   - **Noisy ML Wrapper**: `NoisyCompetitorModelWrapper` introduces controlled error injection to degrade prediction accuracy down to any target level (50% to 90%) for Phase 7 sensitivity benchmarks.

## Common Competitor Interface

```python
get_competitor_price(
    leader_price: float,
    current_follower_price: float,
    context: Optional[Dict[str, Any]] = None,
    mode: str = "rule",
    params: Optional[Dict[str, Any]] = None
) -> Tuple[float, float]
# Returns: (follower_price: float, probability_of_increase: float)
```

## Competitor Response Evaluation (Simulated Data)

> [!WARNING]
> **Data Provenance Notice**: M5 retail data contains NO external competitor prices. All competitor reaction models in mode 'simulated_ml' and 'rule' are derived from calibrated behavioral simulations with known response rules. Never claim competitor behavior was learned from M5 retail data directly.

| Metric | Random Forest (Champion) | Logistic Regression | Hist Gradient Boosting |
| :--- | :---: | :---: | :---: |
| **Validation Accuracy** | **96.40%** | 94.80% | 96.13% |
| **Test Accuracy** | **96.13%** | 94.67% | 95.87% |
| **Test Precision** | **95.73%** | 94.20% | 95.42% |
| **Test Recall** | **96.00%** | 94.57% | 95.71% |
| **Test F1 Score** | **95.86%** | 94.38% | 95.56% |
| **Test ROC AUC** | **0.9945** | 0.9882 | 0.9931 |
| **Brier Score** | **0.0299** | 0.0410 | 0.0315 |

**Behavioral Rule Recovery Metrics**:
- Directional Rule Recovery Rate: **96.13%**
- Probability Recovery MAE: **0.2083**
- Price Magnitude Response MAE: **$0.0504**
- Price Magnitude Response RMSE: **$0.0745**

## Model Comparison (Demand Forecaster - Phase 3)

| Model | Val MAE | Val RMSE | Val WMAPE | Val R² | Test MAE | Test RMSE | Test WMAPE | Test R² |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Hist Gradient Boosting** (Champion) | **3.111** | **5.920** | **35.28%** | **0.8139** | **3.482** | **5.915** | **36.82%** | **0.8081** |
| **Ridge Regression** | 3.306 | 6.182 | 37.48% | 0.7971 | 3.618 | 6.136 | 38.25% | 0.7935 |
| **Linear Regression** | 3.305 | 6.184 | 37.47% | 0.7969 | 3.623 | 6.140 | 38.30% | 0.7932 |
| **Random Forest** | 3.570 | 6.314 | 40.48% | 0.7883 | 3.823 | 6.281 | 40.42% | 0.7836 |
| **Seasonal Naive** (Lag-7 Baseline) | 3.908 | 7.803 | 44.32% | 0.6767 | 4.443 | 7.743 | 46.97% | 0.6712 |

## Interfaces (function signatures)

### `src.competitor.response_simulator`
- `rule_based_response(leader_price: float, current_follower_price: float, context: Optional[Dict[str, Any]] = None, params: Optional[Dict[str, Any]] = None) -> Tuple[float, float]`
- `generate_synthetic_competitor_dataset(processed_df: pd.DataFrame, n_samples: int = 4000, behavior: str = "adaptive", noise_std: float = 0.02, random_seed: int = RANDOM_SEED) -> pd.DataFrame`

### `src.competitor.response_model`
- `get_competitor_price(leader_price: float, current_follower_price: float, context: Optional[Dict[str, Any]] = None, mode: str = "rule", params: Optional[Dict[str, Any]] = None) -> Tuple[float, float]`
- `class CompetitorResponseModel(classifier: Any, regressor: Any, feature_names: List[str], classifier_name: str, regressor_name: str)`
  - `predict_response(features_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]`
  - `predict_single(leader_price: float, current_follower_price: float, context: Optional[Dict[str, Any]] = None) -> Tuple[float, float]`
- `class NoisyCompetitorModelWrapper(base_model: CompetitorResponseModel, target_accuracy: float = 0.70, random_seed: int = RANDOM_SEED)`
  - `predict_single(leader_price: float, current_follower_price: float, context: Optional[Dict[str, Any]] = None) -> Tuple[float, float]`
- `load_real_competitor_data(filepath: Optional[Union[str, Path]] = None) -> pd.DataFrame`
- `train_competitor_models(train_df: pd.DataFrame, val_df: pd.DataFrame, feature_cols: Optional[List[str]] = None, save_path: Optional[Union[str, Path]] = None) -> Tuple[CompetitorResponseModel, Dict[str, Any]]`
- `save_competitor_model(wrapper: CompetitorResponseModel, path: Optional[Union[str, Path]] = None) -> Path`
- `load_competitor_model(path: Optional[Union[str, Path]] = None) -> CompetitorResponseModel`

### `src.competitor.response_evaluation`
- `evaluate_response_classifier(y_true: Union[pd.Series, np.ndarray], y_prob: Union[pd.Series, np.ndarray], threshold: float = 0.50) -> Dict[str, Any]`
- `evaluate_rule_recovery(test_df: pd.DataFrame, predicted_probs: np.ndarray, predicted_prices: np.ndarray) -> Dict[str, Any]`
- `plot_confusion_matrix(cm: List[List[int]], labels: Tuple[str, str] = ("Maintain/Cut", "Increase")) -> go.Figure`
- `plot_calibration_curve(y_true: Union[pd.Series, np.ndarray], y_prob: Union[pd.Series, np.ndarray], n_bins: int = 10) -> go.Figure`
- `plot_rule_recovery(leader_pct_changes: np.ndarray, true_probs: np.ndarray, predicted_probs: np.ndarray) -> go.Figure`

## Known issues

None. Phases 1–4 passing 28/28 unit tests cleanly. Ready for Phase 5.
