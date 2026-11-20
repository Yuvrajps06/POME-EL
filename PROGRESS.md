# PricePilot: Progress Tracker

## Checklist

- [x] Phase 1: Data Pipeline & Preprocessing
- [x] Phase 2: Exploratory Data Analysis & Market Intelligence
- [ ] Phase 3: Demand Modeling & Elasticity Estimation
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
   - **Primary Demonstration Item**: `FOODS_3_090` (highest volume hero item, 8 distinct price points from $1.00 to $1.60, CV = 0.0915, ~66.5 units/day in CA_1, ~29.4 in CA_2, low zero-rate).
2. **Missing Price Treatment**:
   - Rows where `sell_price` is missing correspond to weeks when an item was not yet on shelves or stocked. These rows are dropped rather than imputed.
3. **Leakage Prevention**:
   - Lag features (`lag_sales_1`, `7`, `14`, `28`) use grouped `.shift(k)`.
   - Rolling features (`rolling_mean_7`, `28`, `rolling_std_7`, `28`) apply `.shift(1)` before computing rolling statistics to ensure target day sales is strictly excluded.
4. **Chronological Splitting**:
   - Strict temporal ordering: Test set (last 28 days: 2016-04-25 to 2016-05-22), Validation set (28 days prior: 2016-03-28 to 2016-04-24), Train set (preceding history).
5. **Phase 2 EDA Findings**:
   - **Price Variation**: Items like `FOODS_3_090`, `FOODS_3_723`, and `FOODS_3_491` exhibit 8 distinct price levels with moderate price volatility ($CV \approx 0.08 - 0.10$), making them well-identified for empirical demand curve estimation.
   - **Recommended Modeling Candidates**:
     - *Hero Candidate*: `FOODS_3_090` (clear downward sloping log-log demand curve, heavy promo activity at $1.00 and regular pricing at $1.25-$1.60).
     - *Alternative Candidates*: `FOODS_3_202` (mid-tier price $4.32, 16.4 units/day, 11% zero share) and `FOODS_3_723` ($1.38 avg price, 14.9 units/day).
   - **Data Concerns**:
     - *Intermittent demand*: For lower volume items, zero sales occur on 25-35% of days, reinforcing the requirement for log1p transformation (`np.log1p(sales)`) or count-based demand formulations.
     - *Confounding*: Price drops frequently coincide with SNAP issuance windows (days 1-10 of month) and weekend peaks. Demand models must control for SNAP, day-of-week, and event indicators to avoid omitted variable bias in observational elasticity.

## Output Files & Schema

- `data/processed/m5_processed.parquet` (57,390 rows, 37 columns)
- `data/processed/m5_processed.csv` (57,390 rows, 37 columns)
- `data/sample/m5_sample.csv` (100 rows, 37 columns)
- `notebooks/market_kpis.csv` (Market intelligence KPIs for 30 series)
- `notebooks/run_eda.py` & `notebooks/eda.ipynb` (Demonstration scripts)

**Columns**:
`item_id`, `dept_id`, `cat_id`, `store_id`, `state_id`, `d`, `sales`, `date`, `wm_yr_wk`, `weekday`, `wday`, `month`, `year`, `event_name_1`, `event_type_1`, `snap_CA`, `snap_TX`, `snap_WI`, `sell_price`, `is_outlier`, `price_change_pct`, `price_rel_mean`, `is_discounted`, `lag_sales_1`, `lag_sales_7`, `lag_sales_14`, `lag_sales_28`, `rolling_mean_7`, `rolling_std_7`, `rolling_mean_28`, `rolling_std_28`, `day_of_week`, `day_of_month`, `is_weekend`, `has_event`, `log_price`, `log_sales`.

## Interfaces (function signatures)

### `src.data.loader`
- `load_calendar(path: Optional[Union[str, Path]] = None) -> pd.DataFrame`
- `load_prices(path: Optional[Union[str, Path]] = None, store_ids: Optional[List[str]] = None, item_ids: Optional[List[str]] = None) -> pd.DataFrame`
- `load_sales(path: Optional[Union[str, Path]] = None, category: Optional[str] = None, store_ids: Optional[List[str]] = None, item_ids: Optional[List[str]] = None) -> pd.DataFrame`
- `rank_candidate_items(prices_df: pd.DataFrame, sales_df: pd.DataFrame, stores: Optional[List[str]] = None, top_n: int = 20) -> pd.DataFrame`

### `src.data.preprocessing`
- `melt_sales_to_long(sales_df: pd.DataFrame) -> pd.DataFrame`
- `clean_and_merge(sales_long: pd.DataFrame, calendar_df: pd.DataFrame, prices_df: pd.DataFrame) -> pd.DataFrame`
- `flag_outliers(df: pd.DataFrame, iqr_multiplier: float = 3.0) -> pd.DataFrame`
- `run_preprocessing(category: Optional[str] = DEFAULT_CATEGORY, stores: Optional[List[str]] = None, items: Optional[List[str]] = None) -> pd.DataFrame`

### `src.data.feature_engineering`
- `add_lag_features(df: pd.DataFrame, lags: List[int] = [1, 7, 14, 28]) -> pd.DataFrame`
- `add_rolling_features(df: pd.DataFrame, windows: List[int] = [7, 28]) -> pd.DataFrame`
- `add_price_features(df: pd.DataFrame) -> pd.DataFrame`
- `add_calendar_features(df: pd.DataFrame) -> pd.DataFrame`
- `add_log_transforms(df: pd.DataFrame) -> pd.DataFrame`
- `build_feature_pipeline(df: pd.DataFrame) -> pd.DataFrame`
- `chronological_split(df: pd.DataFrame, test_days: int = 28, val_days: int = 28) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]`

### `src.data.eda`
- `compute_market_kpis(df: pd.DataFrame, item_id: Optional[str] = None, store_id: Optional[str] = None) -> pd.DataFrame`
- `plot_sales_and_price_over_time(df: pd.DataFrame, item_id: str, store_id: str) -> go.Figure`
- `plot_price_vs_quantity(df: pd.DataFrame, item_id: str, store_id: str, log_scale: bool = False, add_trendline: bool = True) -> go.Figure`
- `plot_sales_distribution(df: pd.DataFrame, item_id: Optional[str] = None, store_id: Optional[str] = None) -> go.Figure`
- `plot_seasonality(df: pd.DataFrame, item_id: Optional[str] = None, store_id: Optional[str] = None, freq: str = "dow") -> go.Figure`
- `plot_price_change_vs_sales_change(df: pd.DataFrame, item_id: Optional[str] = None, store_id: Optional[str] = None) -> go.Figure`
- `plot_promotion_effect(df: pd.DataFrame, item_id: Optional[str] = None, store_id: Optional[str] = None) -> go.Figure`
- `plot_product_comparison(df: pd.DataFrame, item_ids: Optional[List[str]] = None, metric: str = "sales") -> go.Figure`

## Known issues

None. Phase 1 and Phase 2 pipelines and tests passing cleanly. Ready for Phase 3.
