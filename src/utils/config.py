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

# Default Placeholders for M5 Subset Filtering
DEFAULT_CATEGORY: str = "FOODS_3"
DEFAULT_DEPARTMENT: str = "FOODS_3"
DEFAULT_STORES: List[str] = ["CA_1", "CA_2"]
DEFAULT_LEADER_STORE: str = "CA_1"
DEFAULT_FOLLOWER_STORE: str = "CA_2"
DEFAULT_ITEMS: List[str] = [
    "FOODS_3_090",
    "FOODS_3_586",
    "FOODS_3_252",
]

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
    stores: List[str] = field(default_factory=lambda: list(DEFAULT_STORES))
    leader_store: str = DEFAULT_LEADER_STORE
    follower_store: str = DEFAULT_FOLLOWER_STORE
    items: List[str] = field(default_factory=lambda: list(DEFAULT_ITEMS))

    price_grid_min_factor: float = PRICE_GRID_MIN_FACTOR
    price_grid_max_factor: float = PRICE_GRID_MAX_FACTOR
    price_grid_num_steps: int = PRICE_GRID_NUM_STEPS
    price_grid_min_price: float = PRICE_GRID_MIN_PRICE
    default_cost_ratio: float = DEFAULT_UNIT_COST_RATIO


# Default config instance
config = Config()
