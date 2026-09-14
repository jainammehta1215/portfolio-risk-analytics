"""
Configuration for the risk & performance dashboard.

A portfolio is a dict of ticker -> weight. Weights are normalised to sum to
one. The benchmark may be a single ticker or another weighted dict (for
example a 60/40 blend), which is treated exactly like a portfolio.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd

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


# --------------------------------------------------------------------------- #
# Market presets
# --------------------------------------------------------------------------- #
# One word in the notebook (MARKET = "IN") sets the base currency, a local
# risk-free series from FRED, a sensible default benchmark and stress windows
# that matter for that market. Global shocks are shared across presets.
_GLOBAL: Dict[str, tuple] = {
    "Q4 2018 sell-off": ("2018-09-20", "2018-12-24"),
    "Covid crash (2020)": ("2020-02-19", "2020-03-23"),
    "Rate shock (2022)": ("2022-01-03", "2022-10-12"),
    "Aug 2024 yen-carry unwind": ("2024-07-16", "2024-08-05"),
}


@dataclass
class MarketPreset:
    code: str
    name: str
    currency: str
    fred_series: str                    # short-term rate, percent p.a.
    rf_note: str
    benchmark: Dict[str, float]
    benchmark_name: str
    benchmark_start: str                # earliest date the benchmark exists
    scenarios: Dict[str, tuple]
    suffix_hint: str = ""

    def data_config(self, base: DataConfig) -> DataConfig:
        """Copy of `base` with this market's currency, risk-free series and a start no earlier than the benchmark."""
        start = max(pd.Timestamp(base.start), pd.Timestamp(self.benchmark_start)).strftime("%Y-%m-%d")
        return DataConfig(start=start, end=base.end, cache_dir=base.cache_dir, fred_series=self.fred_series,
                          base_currency=self.currency, trading_days=base.trading_days,
                          min_history_fraction=base.min_history_fraction)


MARKETS: Dict[str, MarketPreset] = {
    "US": MarketPreset("US", "United States", "USD", "DGS3MO", "3-month T-bill (daily)",
                       {"SPY": 0.60, "AGG": 0.40}, "60/40 (SPY/AGG)", "2010-01-01",
                       STRESS_SCENARIOS, ""),
    "IN": MarketPreset("IN", "India", "INR", "IRSTCI01INM156N", "RBI call money rate (monthly)",
                       {"NIFTYBEES.NS": 1.0}, "Nifty 50 (NIFTYBEES)", "2010-01-01", {
                           "Taper tantrum / rupee crisis (2013)": ("2013-05-22", "2013-08-28"),
                           "Global growth scare (2015-16)": ("2015-08-10", "2016-02-11"),
                           "Demonetisation (Nov 2016)": ("2016-11-08", "2016-12-26"),
                           "IL&FS credit crisis (2018)": ("2018-08-28", "2018-10-26"),
                           "Covid crash (2020)": ("2020-01-17", "2020-03-23"),
                           "Rate shock / FPI outflows (2022)": ("2022-01-17", "2022-06-17"),
                           "Hindenburg-Adani (2023)": ("2023-01-24", "2023-02-22"),
                           "Election result day (4 Jun 2024)": ("2024-06-03", "2024-06-04"),
                           "PSU / mid-cap correction (Sep 24 - Mar 25)": ("2024-09-26", "2025-03-03"),
                       }, "NSE symbols end in .NS (RELIANCE.NS); BSE in .BO"),
    "UK": MarketPreset("UK", "United Kingdom", "GBP", "IR3TIB01GBM156N", "3-month interbank (monthly)",
                       {"ISF.L": 1.0}, "FTSE 100 (ISF)", "2010-01-01", {
                           "Euro crisis (2011)": ("2011-07-22", "2011-10-03"),
                           "Brexit referendum (2016)": ("2016-06-23", "2016-06-27"),
                           "Mini-budget gilt crisis (2022)": ("2022-09-22", "2022-10-12"),
                           **_GLOBAL}, "LSE symbols end in .L (HSBA.L); prices quoted in pence are handled"),
    "EU": MarketPreset("EU", "Euro area", "EUR", "IR3TIB01EZM156N", "3-month Euribor (monthly)",
                       {"EXW1.DE": 1.0}, "Euro Stoxx 50 (EXW1)", "2010-01-01", {
                           "Euro sovereign crisis (2011)": ("2011-07-22", "2011-09-12"),
                           "Brexit referendum (2016)": ("2016-06-23", "2016-06-27"),
                           "Ukraine invasion (2022)": ("2022-02-16", "2022-03-08"),
                           **_GLOBAL}, "Xetra .DE, Paris .PA, Milan .MI, Amsterdam .AS"),
    "JP": MarketPreset("JP", "Japan", "JPY", "IR3TIB01JPM156N", "3-month interbank (monthly)",
                       {"1321.T": 1.0}, "Nikkei 225 (1321)", "2010-01-01", {
                           "Tohoku earthquake (2011)": ("2011-03-10", "2011-03-15"),
                           "Taper tantrum (2013)": ("2013-05-22", "2013-06-13"),
                           "China deval / yen surge (2015-16)": ("2015-08-10", "2016-02-12"),
                           **_GLOBAL}, "Tokyo symbols are numeric with .T (7203.T)"),
    "AE": MarketPreset("AE", "United Arab Emirates", "USD", "DGS3MO", "US 3-month T-bill (AED is pegged to USD)",
                       {"UAE": 1.0}, "MSCI UAE (iShares UAE ETF)", "2014-05-01", {
                           "Oil price collapse (2014-15)": ("2014-09-01", "2015-01-30"),
                           "Oil / China scare (2015-16)": ("2015-08-10", "2016-01-21"),
                           "Covid + oil war (2020)": ("2020-02-19", "2020-03-23"),
                           **{k: v for k, v in _GLOBAL.items() if "Covid" not in k}},
                       "DFM/ADX symbols end in .AE (EMAAR.AE); the benchmark ETF starts in 2014"),
}


def market_preset(code: str) -> MarketPreset:
    try:
        return MARKETS[code.upper()]
    except KeyError:
        raise KeyError(f"Unknown market '{code}'. Available: {', '.join(MARKETS)}") from None
