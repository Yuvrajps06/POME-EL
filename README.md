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
# Install dependencies
pip install -r requirements.txt

# Run tests
pytest

# Launch Streamlit dashboard
streamlit run app.py
```
