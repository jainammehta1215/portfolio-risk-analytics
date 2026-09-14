"""
Data acquisition layer (shared across the asset-allocation track).

Responsibilities
----------------
* Download adjusted prices from Yahoo Finance with local caching.
* Download the risk-free rate from FRED (no API key required).
* Build a point-in-time market-cap proxy (current shares outstanding x
  historical price) for Black-Litterman equilibrium weights.
* Run data-quality checks and fail loudly on problems that would corrupt
  downstream estimates.
* Provide a synthetic fallback so the pipeline can be unit-tested offline.

Every public function returns pandas objects indexed by a tz-naive
DatetimeIndex and aligned to the trading calendar of the price data.
"""
from __future__ import annotations

import io
import logging
import os
import time
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

from .config import DATA, DataConfig

logger = logging.getLogger(__name__)

FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _cache_path(cfg: DataConfig, name: str) -> str:
    os.makedirs(cfg.cache_dir, exist_ok=True)
    return os.path.join(cfg.cache_dir, name)


def _retry(fn, attempts: int = 3, wait: float = 2.0, label: str = "call"):
    """Retry a callable with linear back-off. Re-raises the last exception."""
    last_exc: Optional[Exception] = None
    for i in range(1, attempts + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - we want to retry on anything
            last_exc = exc
            logger.warning("%s failed (attempt %d/%d): %s", label, i, attempts, exc)
            time.sleep(wait * i)
    assert last_exc is not None
    raise last_exc


# --------------------------------------------------------------------------- #
# Prices
# --------------------------------------------------------------------------- #
def load_prices(tickers: Iterable[str],
                cfg: DataConfig = DATA,
                force_refresh: bool = False) -> pd.DataFrame:
    """
    Adjusted close prices, one column per ticker.

    Prices are cached to CSV keyed on the ticker set and date range.
    Adjusted closes (auto_adjust=True) already incorporate splits and
    dividends, so simple returns on them are total returns.
    """
    tickers = sorted(set(tickers))
    end = cfg.end or pd.Timestamp.today().strftime("%Y-%m-%d")
    cache = _cache_path(cfg, f"prices_{cfg.start}_{end}_{len(tickers)}.csv")

    if os.path.exists(cache) and not force_refresh:
        prices = pd.read_csv(cache, index_col=0, parse_dates=True)
        if set(prices.columns) == set(tickers):
            logger.info("Loaded %d tickers from cache %s", len(tickers), cache)
            return prices[tickers]

    import yfinance as yf  # imported lazily so tests can run without it

    def _download() -> pd.DataFrame:
        raw = yf.download(tickers, start=cfg.start, end=end,
                          auto_adjust=True, progress=False, threads=True)
        if raw.empty:
            raise RuntimeError("Yahoo Finance returned an empty frame")
        close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
        if isinstance(close, pd.Series):
            close = close.to_frame(tickers[0])
        return close

    prices = _retry(_download, label="yfinance download")
    prices.index = pd.to_datetime(prices.index).tz_localize(None)
    prices = prices.sort_index()
    prices.columns = [str(c) for c in prices.columns]
    prices = prices.reindex(columns=tickers)

    try:
        prices.to_csv(cache)
    except Exception as exc:  # read-only filesystem etc. - caching is best-effort
        logger.warning("Could not write price cache: %s", exc)
    return prices


def repair_bad_prints(prices: pd.DataFrame, threshold: float = 0.5,
                      lookahead: int = 5, tol: float = 0.2) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Blank out price spikes that reverse within `lookahead` days.

    A genuine crash does not un-crash in a week; an unadjusted split does not
    reverse at all. Only the self-reversing pattern is treated as an error.
    Returns the repaired frame and a per-ticker count of repaired prints.
    """
    px = prices.copy()
    fixed: Dict[str, int] = {}
    for t in px.columns:
        s = px[t]
        r = s.pct_change()
        bad = r.index[r.abs() > threshold]
        for d in bad:
            i = s.index.get_loc(d)
            base = s.iloc[i - 1]
            if not np.isfinite(base) or base <= 0:
                continue
            window = s.iloc[i: i + lookahead + 1]
            back = (window / base - 1).abs()
            hits = np.where(back.values[1:] < tol)[0]
            if len(hits):
                j = i + 1 + hits[0]                      # first index back near the base
                px.loc[s.index[i]: s.index[j - 1], t] = np.nan
                fixed[t] = fixed.get(t, 0) + (j - i)
    return px, fixed


def clean_prices(prices: pd.DataFrame,
                 cfg: DataConfig = DATA) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Data-quality gate.

    * Drops tickers with insufficient history.
    * Forward-fills isolated gaps (max 5 days) and reports them.
    * Drops any leading rows where not every ticker has a price.
    * Raises on non-positive prices or on duplicated dates.

    Returns the cleaned frame and a dict of issues found, for the report.
    """
    issues: Dict[str, str] = {}
    px = prices.copy()

    if px.index.duplicated().any():
        raise ValueError("Duplicated dates in price index")

    # 1. Cross-exchange calendars: drop days on which most of the universe was
    #    closed (e.g. an Indian holiday in a mixed India/US list). Dropping a
    #    row is safe for returns because pct_change compounds across the gap;
    #    forward-filling instead would create artificial zero-return days.
    row_cov = px.notna().mean(axis=1)
    dropped_days = int((row_cov < 0.5).sum())
    if dropped_days:
        issues["_calendar"] = f"dropped {dropped_days} days when <50% of tickers traded"
    px = px[row_cov >= 0.5]

    # 2. Repair self-reversing bad prints: a move beyond +/-50% that is undone
    #    within 5 days (cumulative move back within 20% of the start) is a data
    #    error, not a market event. Blank the bad prices so step 3 fills them.
    px, glitches = repair_bad_prints(px)
    for t, n in glitches.items():
        issues[t] = issues.get(t, "") + f"repaired {n} bad print(s); "

    # 3. Fill the remaining isolated gaps (single-exchange holidays, bad ticks).
    gap_counts = px.isna().sum()
    px = px.ffill(limit=5)

    # 4. Tickers with insufficient history (listed after `start`, delisted, etc).
    coverage = px.notna().mean()
    thin = coverage[coverage < cfg.min_history_fraction]
    for t, c in thin.items():
        first = px[t].first_valid_index()
        issues[t] = (f"dropped: only {c:.1%} of dates have a price"
                     + (f" (first price {first.date()}; move DataConfig.start later to keep it)" if first is not None else ""))
    px = px.drop(columns=thin.index)
    for t, n in gap_counts.reindex(px.columns).items():
        if n > 0:
            issues[t] = issues.get(t, "") + f"filled {int(n)} missing price(s)"

    px = px.dropna(how="any")          # remove leading NaNs / anything unfillable

    if (px <= 0).any().any():
        bad = px.columns[(px <= 0).any()].tolist()
        raise ValueError(f"Non-positive prices found for {bad}")

    return px, issues


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Simple daily returns. Simple (not log) because portfolio returns aggregate linearly."""
    rets = prices.pct_change().iloc[1:]
    # Guard against corporate-action glitches that produce absurd single-day moves.
    extreme = (rets.abs() > 0.5)
    if extreme.any().any():
        n = int(extreme.sum().sum())
        logger.warning("%d daily returns exceed +/-50%%; inspect before trusting results", n)
    return rets


# --------------------------------------------------------------------------- #
# Currency conversion
# --------------------------------------------------------------------------- #
def fetch_currencies(tickers: Iterable[str]) -> pd.Series:
    """Quote currency per ticker from yfinance (e.g. USD, INR, JPY, GBp)."""
    out: Dict[str, str] = {}
    try:
        import yfinance as yf
        for t in tickers:
            try:
                out[t] = str(yf.Ticker(t).fast_info.get("currency") or "USD")
            except Exception as exc:  # noqa: BLE001
                logger.warning("currency lookup failed for %s; assuming USD: %s", t, exc)
                out[t] = "USD"
    except ImportError:
        out = {t: "USD" for t in tickers}
    return pd.Series(out, name="currency")


def convert_to_base_currency(prices: pd.DataFrame, base: str = "USD",
                             currencies: Optional[pd.Series] = None,
                             cfg: DataConfig = DATA) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Convert every column to `base` using Yahoo FX crosses (e.g. INRUSD=X).

    GBp (pence) and ZAc (cents) are scaled to their major unit first. FX
    series are forward-filled onto the price calendar. Returns the converted
    frame and a log of what was converted. No-op for single-currency
    universes already quoted in `base`.
    """
    log: Dict[str, str] = {}
    cur = currencies if currencies is not None else fetch_currencies(prices.columns)
    cur = cur.reindex(prices.columns).fillna(base)
    px = prices.copy()

    minor = {"GBp": ("GBP", 100.0), "ZAc": ("ZAR", 100.0), "ILA": ("ILS", 100.0)}
    for t, c in cur.items():
        if c in minor:
            major, div = minor[c]
            px[t] = px[t] / div
            cur[t] = major

    needed = sorted(set(cur) - {base})
    if not needed:
        return px, log

    import yfinance as yf
    end = cfg.end or pd.Timestamp.today().strftime("%Y-%m-%d")
    for c in needed:
        pair = f"{c}{base}=X"
        def _dl(pair=pair):
            fx = yf.download(pair, start=cfg.start, end=end, auto_adjust=True, progress=False)
            if fx.empty:
                raise RuntimeError(f"no FX data for {pair}")
            fx = fx["Close"]
            return fx.iloc[:, 0] if isinstance(fx, pd.DataFrame) else fx
        try:
            fx = _retry(_dl, label=f"FX {pair}")
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Could not download {pair}; cannot express {c} assets in {base}") from exc
        fx.index = pd.to_datetime(fx.index).tz_localize(None)
        fx = fx.reindex(px.index.union(fx.index)).ffill().reindex(px.index)
        cols = [t for t, cc in cur.items() if cc == c]
        px[cols] = px[cols].mul(fx, axis=0)
        log[c] = f"{len(cols)} ticker(s) converted via {pair}"
    return px, log


# --------------------------------------------------------------------------- #
# Risk-free rate
# --------------------------------------------------------------------------- #
def load_risk_free(index: pd.DatetimeIndex,
                   cfg: DataConfig = DATA,
                   force_refresh: bool = False) -> pd.Series:
    """
    Daily risk-free rate aligned to `index`, derived from a FRED yield series
    quoted in percent (annualised, bond-equivalent). Falls back to zero with a
    loud warning if FRED is unreachable, so the pipeline never silently uses a
    stale number.
    """
    cache = _cache_path(cfg, f"fred_{cfg.fred_series}.csv")
    try:
        if os.path.exists(cache) and not force_refresh:
            raw = pd.read_csv(cache, index_col=0, parse_dates=True)
        else:
            resp = _retry(lambda: requests.get(FRED_CSV_URL.format(series=cfg.fred_series), timeout=30),
                          label="FRED download")
            resp.raise_for_status()
            raw = pd.read_csv(io.StringIO(resp.text), index_col=0, parse_dates=True)
            raw.to_csv(cache)
        yld = pd.to_numeric(raw.iloc[:, 0], errors="coerce")
    except Exception as exc:  # noqa: BLE001
        logger.error("Risk-free download failed (%s). Using 0%% - Sharpe ratios will be overstated.", exc)
        return pd.Series(0.0, index=index, name="rf_daily")

    yld = yld.reindex(index.union(yld.index)).ffill().reindex(index)
    yld = yld.fillna(0.0)
    daily = (1.0 + yld / 100.0) ** (1.0 / cfg.trading_days) - 1.0
    daily.name = "rf_daily"
    return daily


# --------------------------------------------------------------------------- #
# Synthetic fallback (offline tests / demos)
# --------------------------------------------------------------------------- #
def synthetic_prices(tickers: List[str],
                     n_days: int = 2520,
                     seed: int = 7,
                     start: str = "2015-01-01") -> pd.DataFrame:
    """
    Geometric Brownian motion with a one-factor correlation structure.
    Deterministic given `seed`. Used only when live data is unavailable.
    """
    rng = np.random.default_rng(seed)
    n = len(tickers)
    beta = rng.uniform(0.6, 1.4, n)
    idio = rng.uniform(0.15, 0.35, n) / np.sqrt(252)
    mkt = rng.normal(0.0004, 0.011, n_days)
    eps = rng.normal(0, 1, (n_days, n)) * idio
    rets = np.outer(mkt, beta) + eps + rng.uniform(0.0, 0.0003, n)
    idx = pd.bdate_range(start, periods=n_days)
    return pd.DataFrame(100 * np.cumprod(1 + rets, axis=0), index=idx, columns=tickers)


def load_all(tickers: Iterable[str], cfg: DataConfig = DATA,
             use_synthetic: bool = False) -> Dict[str, object]:
    """
    One-call loader: prices (base currency), daily returns, daily risk-free,
    and an issues log. `tickers` should include the benchmark constituents.
    """
    tickers = list(dict.fromkeys(tickers))          # de-duplicate, keep order
    if use_synthetic:
        logger.warning("Using SYNTHETIC prices - results are illustrative only")
        prices = synthetic_prices(tickers)
        issues: Dict[str, str] = {"_mode": "synthetic"}
        returns = compute_returns(prices)
        rf = pd.Series(0.02 / cfg.trading_days, index=returns.index, name="rf_daily")
    else:
        prices = load_prices(tickers, cfg)
        prices, issues = clean_prices(prices, cfg)
        prices, fx_log = convert_to_base_currency(prices, cfg.base_currency, cfg=cfg)
        prices = prices.dropna(how="any")
        issues.update({f"_fx_{k}": v for k, v in fx_log.items()})
        returns = compute_returns(prices)
        rf = load_risk_free(returns.index, cfg)
    return dict(prices=prices, returns=returns, rf=rf, issues=issues)
