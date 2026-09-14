"""
Configuration for the risk & performance dashboard.

A portfolio is a dict of ticker -> weight. Weights are normalised to sum to
one. The benchmark may be a single ticker or another weighted dict (for
example a 60/40 blend), which is treated exactly like a portfolio.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

# --------------------------------------------------------------------------- #
# Default portfolio: a diversified multi-asset ETF allocation
# --------------------------------------------------------------------------- #
PORTFOLIO: Dict[str, float] = {
    "SPY": 0.30,   # US large cap
    "QQQ": 0.10,   # US growth / tech
    "EFA": 0.10,   # developed ex-US equity
    "EEM": 0.05,   # emerging markets equity
    "IEF": 0.15,   # 7-10y Treasuries
    "TLT": 0.10,   # 20y+ Treasuries
    "LQD": 0.05,   # investment-grade credit
    "GLD": 0.10,   # gold
    "VNQ": 0.05,   # US REITs
}

BENCHMARK: Dict[str, float] = {"SPY": 0.60, "AGG": 0.40}      # classic 60/40
BENCHMARK_NAME: str = "60/40 (SPY/AGG)"

# Asset-class grouping used for exposure and risk aggregation.
ASSET_CLASS: Dict[str, str] = {
    "SPY": "Equity - US", "QQQ": "Equity - US", "EFA": "Equity - Intl", "EEM": "Equity - EM",
    "IEF": "Rates", "TLT": "Rates", "AGG": "Rates", "LQD": "Credit",
    "GLD": "Commodities", "VNQ": "Real Estate",
}

# Historical stress windows (start, end) used for scenario analysis.
STRESS_SCENARIOS: Dict[str, tuple] = {
    "Euro crisis / US downgrade (2011)": ("2011-07-22", "2011-10-03"),
    "Taper tantrum (2013)": ("2013-05-02", "2013-06-24"),
    "China deval / vol spike (2015)": ("2015-08-10", "2015-08-25"),
    "Q4 2018 sell-off": ("2018-09-20", "2018-12-24"),
    "Covid crash (2020)": ("2020-02-19", "2020-03-23"),
    "Rate shock (2022)": ("2022-01-03", "2022-10-12"),
    "SVB / regional banks (2023)": ("2023-03-08", "2023-03-17"),
    "Aug 2024 yen-carry unwind": ("2024-07-16", "2024-08-05"),
}


@dataclass
class DataConfig:
    start: str = "2010-01-01"
    end: Optional[str] = None
    cache_dir: str = "data/cache"
    fred_series: str = "DGS3MO"
    base_currency: str = "USD"
    trading_days: int = 252
    min_history_fraction: float = 0.98


@dataclass
class DashboardConfig:
    rebalance: Optional[str] = "ME"     # None = buy-and-hold drift; "ME" monthly; "QE" quarterly
    rolling_window: int = 126           # ~6 months for rolling stats
    long_window: int = 756              # ~3 years for rolling Sharpe
    var_level: float = 0.95
    top_drawdowns: int = 5
    cov_window: int = 252               # window for the risk decomposition covariance


DATA = DataConfig()
DASH = DashboardConfig()
