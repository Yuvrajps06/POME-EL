"""PricePilot central configuration module.

Defines repository paths, global seeds, default selection placeholders,
and price grid search boundaries.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

# Base Paths
ROOT_DIR: Path = Path(__file__).resolve().parent.parent.parent
DATA_DIR: Path = ROOT_DIR / "data"
DATA_RAW_DIR: Path = DATA_DIR / "raw" / "m5"
DATA_PROCESSED_DIR: Path = DATA_DIR / "processed"
DATA_SAMPLE_DIR: Path = DATA_DIR / "sample"
MODELS_DIR: Path = ROOT_DIR / "models"
SRC_DIR: Path = ROOT_DIR / "src"
DASHBOARD_DIR: Path = ROOT_DIR / "dashboard"
TESTS_DIR: Path = ROOT_DIR / "tests"

# Global Random Seed
RANDOM_SEED: int = 42

# Selected Subset for Analysis and Modeling
# Store CA_1 (Leader) and CA_2 (Follower/Competitor), Category FOODS / FOODS_3
DEFAULT_CATEGORY: str = "FOODS"
DEFAULT_DEPARTMENT: str = "FOODS_3"
DEFAULT_STORES: List[str] = ["CA_1", "CA_2"]
DEFAULT_LEADER_STORE: str = "CA_1"
DEFAULT_FOLLOWER_STORE: str = "CA_2"

# 15 High-variance, low-intermittency items continuously sold in CA_1 and CA_2
DEFAULT_ITEMS: List[str] = [
    "FOODS_3_090",  # High volume, 8 distinct prices, low zero-rate (hero item)
    "FOODS_3_473",  # Lowest zero-sales share (7.5%), 6 distinct prices
    "FOODS_3_723",  # 8 distinct prices, high daily sales (11.8)
    "FOODS_3_491",  # 8 distinct prices, high daily sales (11.8)
    "FOODS_3_734",  # 7 distinct prices, 18.3% zero-rate
    "FOODS_3_156",  # 6 distinct prices, 11.6% zero-rate
    "FOODS_3_202",  # 5-6 distinct prices, 12.5 daily sales
    "FOODS_3_785",  # 5-6 distinct prices, 8.5 daily sales
    "FOODS_3_324",  # 5-6 distinct prices, 5.0 daily sales
    "FOODS_3_124",  # 5 distinct prices, premium price point ($5.00-$6.00)
    "FOODS_3_227",  # 5 distinct prices, 23.0% zero-rate
    "FOODS_3_755",  # 6 distinct prices, 9.1 daily sales
    "FOODS_3_351",  # 6 distinct prices, 7.8 daily sales
    "FOODS_3_534",  # 6 distinct prices, 3.3 daily sales
    "FOODS_3_704",  # 6 distinct prices, 15.7% zero-rate
]

# Primary Item for Focused Modeling Demonstrations
PRIMARY_MODELING_ITEM: str = "FOODS_3_090"

# Price Grid & Optimization Settings
PRICE_GRID_MIN_FACTOR: float = 0.70  # Min price bound: 70% of baseline price
PRICE_GRID_MAX_FACTOR: float = 1.30  # Max price bound: 130% of baseline price
PRICE_GRID_NUM_STEPS: int = 30       # Number of discrete price evaluation points
PRICE_GRID_MIN_PRICE: float = 0.10   # Absolute minimum price threshold

# Cost Scenario Parameter (M5 has no cost data)
DEFAULT_UNIT_COST_RATIO: float = 0.60  # Default assumed cost = 60% of baseline price


@dataclass
class Config:
    """Consolidated configuration container."""

    root_dir: Path = ROOT_DIR
    data_dir: Path = DATA_DIR
    data_raw_dir: Path = DATA_RAW_DIR
    data_processed_dir: Path = DATA_PROCESSED_DIR
    data_sample_dir: Path = DATA_SAMPLE_DIR
    models_dir: Path = MODELS_DIR

    random_seed: int = RANDOM_SEED

    category: str = DEFAULT_CATEGORY
    department: str = DEFAULT_DEPARTMENT
    stores: List[str] = field(default_factory=lambda: list(DEFAULT_STORES))
    leader_store: str = DEFAULT_LEADER_STORE
    follower_store: str = DEFAULT_FOLLOWER_STORE
    items: List[str] = field(default_factory=lambda: list(DEFAULT_ITEMS))
    primary_item: str = PRIMARY_MODELING_ITEM

    price_grid_min_factor: float = PRICE_GRID_MIN_FACTOR
    price_grid_max_factor: float = PRICE_GRID_MAX_FACTOR
    price_grid_num_steps: int = PRICE_GRID_NUM_STEPS
    price_grid_min_price: float = PRICE_GRID_MIN_PRICE
    default_cost_ratio: float = DEFAULT_UNIT_COST_RATIO


# Default config instance
config = Config()
