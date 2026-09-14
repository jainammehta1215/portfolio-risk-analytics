"""
Performance and risk metrics for daily return series.

Conventions
-----------
* 252 trading days per year.
* Sharpe / Sortino / alpha use the *daily* risk-free series, not a constant.
* Drawdowns are computed on compounded wealth.
* Benchmark-relative statistics (beta, alpha, tracking error, information
  ratio, capture ratios) take an aligned benchmark return series.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

TRADING_DAYS = 252


# --------------------------------------------------------------------------- #
# Basic return / risk
# --------------------------------------------------------------------------- #
def cagr(r: pd.Series, periods: int = TRADING_DAYS) -> float:
    r = r.dropna()
    if len(r) == 0:
        return np.nan
    return float((1 + r).prod()) ** (periods / len(r)) - 1.0


def total_return(r: pd.Series) -> float:
    return float((1 + r.dropna()).prod() - 1.0)


def ann_vol(r: pd.Series, periods: int = TRADING_DAYS) -> float:
    return float(r.std(ddof=1) * np.sqrt(periods))


def downside_deviation(r: pd.Series, rf: Optional[pd.Series] = None,
                       periods: int = TRADING_DAYS) -> float:
    ex = _excess(r, rf)
    return float(np.sqrt(np.mean(np.minimum(ex, 0.0) ** 2)) * np.sqrt(periods))


def _excess(r: pd.Series, rf: Optional[pd.Series]) -> pd.Series:
    return r - (rf.reindex(r.index).fillna(0.0) if rf is not None else 0.0)


def sharpe(r: pd.Series, rf: Optional[pd.Series] = None, periods: int = TRADING_DAYS) -> float:
    ex = _excess(r, rf)
    sd = ex.std(ddof=1)
    return float(ex.mean() / sd * np.sqrt(periods)) if sd > 0 else np.nan


def sortino(r: pd.Series, rf: Optional[pd.Series] = None, periods: int = TRADING_DAYS) -> float:
    ex = _excess(r, rf)
    dd = np.sqrt(np.mean(np.minimum(ex, 0.0) ** 2))
    return float(ex.mean() / dd * np.sqrt(periods)) if dd > 0 else np.nan


def drawdown_series(r: pd.Series) -> pd.Series:
    wealth = (1 + r).cumprod()
    return wealth / wealth.cummax() - 1.0


def max_drawdown(r: pd.Series) -> float:
    return float(drawdown_series(r).min())


def calmar(r: pd.Series, periods: int = TRADING_DAYS) -> float:
    mdd = max_drawdown(r)
    return float(cagr(r, periods) / abs(mdd)) if mdd < 0 else np.nan


def var_cvar(r: pd.Series, level: float = 0.95) -> tuple[float, float]:
    """Historical daily VaR and CVaR at `level`, reported as positive losses."""
    q = r.quantile(1 - level)
    return float(-q), float(-r[r <= q].mean())


def tail_stats(r: pd.Series) -> Dict[str, float]:
    r = r.dropna()
    return dict(skew=float(stats.skew(r)), excess_kurtosis=float(stats.kurtosis(r)),
                best_day=float(r.max()), worst_day=float(r.min()),
                pct_positive_days=float((r > 0).mean()))


def omega_ratio(r: pd.Series, threshold_daily: float = 0.0) -> float:
    ex = r - threshold_daily
    gains, losses = ex[ex > 0].sum(), -ex[ex < 0].sum()
    return float(gains / losses) if losses > 0 else np.nan


# --------------------------------------------------------------------------- #
# Drawdown table
# --------------------------------------------------------------------------- #
def drawdown_table(r: pd.Series, top: int = 5) -> pd.DataFrame:
    """
    The `top` deepest drawdowns with peak, trough and recovery dates, depth,
    days peak-to-trough, days trough-to-recovery, and total length.
    An unrecovered drawdown has NaT recovery and its length runs to the end.
    """
    wealth = (1 + r).cumprod()
    peak = wealth.cummax()
    dd = wealth / peak - 1.0
    in_dd = dd < 0
    episodes = []
    start = None
    for i, (date, flag) in enumerate(in_dd.items()):
        if flag and start is None:
            start = i - 1 if i > 0 else 0           # last peak date
        elif not flag and start is not None:
            episodes.append((start, i))
            start = None
    if start is not None:
        episodes.append((start, None))

    rows = []
    for s, e in episodes:
        seg = dd.iloc[s: e if e is not None else len(dd)]
        trough_idx = seg.idxmin()
        rows.append(dict(peak=dd.index[s], trough=trough_idx,
                         recovery=dd.index[e] if e is not None else pd.NaT,
                         depth=float(seg.min()),
                         days_to_trough=int(dd.index.get_loc(trough_idx) - s),
                         days_to_recover=(int(e - dd.index.get_loc(trough_idx)) if e is not None else np.nan),
                         total_days=int((e if e is not None else len(dd)) - s)))
    if not rows:
        return pd.DataFrame(columns=["peak", "trough", "recovery", "depth", "days_to_trough",
                                     "days_to_recover", "total_days"])
    tbl = pd.DataFrame(rows).sort_values("depth").head(top).reset_index(drop=True)
    return tbl


# --------------------------------------------------------------------------- #
# Calendar tables
# --------------------------------------------------------------------------- #
def monthly_returns(r: pd.Series) -> pd.Series:
    return (1 + r).resample("ME").prod() - 1


def monthly_table(r: pd.Series) -> pd.DataFrame:
    """Year x month table of returns with a 'Year' total column."""
    m = monthly_returns(r)
    tbl = pd.DataFrame({"year": m.index.year, "month": m.index.month, "ret": m.values})
    piv = tbl.pivot(index="year", columns="month", values="ret")
    piv.columns = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][:len(piv.columns)] \
        if len(piv.columns) == 12 else [pd.Timestamp(2000, c, 1).strftime("%b") for c in piv.columns]
    piv["Year"] = (1 + piv.fillna(0)).prod(axis=1) - 1
    return piv


def annual_returns(r: pd.Series) -> pd.Series:
    return (1 + r).groupby(r.index.year).prod() - 1


# --------------------------------------------------------------------------- #
# Benchmark-relative
# --------------------------------------------------------------------------- #
def beta_alpha(r: pd.Series, bench: pd.Series, rf: Optional[pd.Series] = None,
               periods: int = TRADING_DAYS) -> Dict[str, float]:
    """CAPM beta and annualised Jensen alpha via OLS on excess returns."""
    df = pd.concat([_excess(r, rf), _excess(bench, rf)], axis=1).dropna()
    x, y = df.iloc[:, 1].values, df.iloc[:, 0].values
    slope, intercept, rvalue, pvalue, stderr = stats.linregress(x, y)
    resid_se = np.std(y - slope * x - intercept, ddof=2) / np.sqrt(len(x))
    return dict(beta=float(slope), alpha_ann=float(intercept * periods),
                r_squared=float(rvalue ** 2), beta_se=float(stderr),
                alpha_t=float(intercept / resid_se) if resid_se > 0 else np.nan)


def tracking_error(r: pd.Series, bench: pd.Series, periods: int = TRADING_DAYS) -> float:
    active = (r - bench.reindex(r.index)).dropna()
    return float(active.std(ddof=1) * np.sqrt(periods))


def information_ratio(r: pd.Series, bench: pd.Series, periods: int = TRADING_DAYS) -> float:
    active = (r - bench.reindex(r.index)).dropna()
    sd = active.std(ddof=1)
    return float(active.mean() / sd * np.sqrt(periods)) if sd > 0 else np.nan


def capture_ratios(r: pd.Series, bench: pd.Series, freq: str = "ME") -> Dict[str, float]:
    """Up/down capture on period-aggregated returns (monthly by default)."""
    p = (1 + r).resample(freq).prod() - 1
    b = (1 + bench.reindex(r.index)).resample(freq).prod() - 1
    up, down = b > 0, b < 0
    def _geo(x):
        return float((1 + x).prod() ** (1 / max(len(x), 1)) - 1) if len(x) else np.nan
    up_c = _geo(p[up]) / _geo(b[up]) if up.any() and _geo(b[up]) != 0 else np.nan
    down_c = _geo(p[down]) / _geo(b[down]) if down.any() and _geo(b[down]) != 0 else np.nan
    return dict(up_capture=up_c, down_capture=down_c,
                capture_ratio=(up_c / down_c if down_c not in (0, np.nan) and not np.isnan(down_c) else np.nan),
                hit_rate=float((p > b).mean()))


def correlation(r: pd.Series, bench: pd.Series) -> float:
    return float(pd.concat([r, bench], axis=1).dropna().corr().iloc[0, 1])


# --------------------------------------------------------------------------- #
# Rolling statistics
# --------------------------------------------------------------------------- #
def rolling_vol(r: pd.Series, window: int, periods: int = TRADING_DAYS) -> pd.Series:
    return r.rolling(window).std(ddof=1) * np.sqrt(periods)


def rolling_sharpe(r: pd.Series, rf: Optional[pd.Series], window: int,
                   periods: int = TRADING_DAYS) -> pd.Series:
    ex = _excess(r, rf)
    return ex.rolling(window).mean() / ex.rolling(window).std(ddof=1) * np.sqrt(periods)


def rolling_beta(r: pd.Series, bench: pd.Series, window: int) -> pd.Series:
    df = pd.concat([r, bench.reindex(r.index)], axis=1).dropna()
    cov = df.iloc[:, 0].rolling(window).cov(df.iloc[:, 1])
    var = df.iloc[:, 1].rolling(window).var()
    return cov / var


def rolling_correlation(r: pd.Series, bench: pd.Series, window: int) -> pd.Series:
    return r.rolling(window).corr(bench.reindex(r.index))


def rolling_drawdown(r: pd.Series, window: int) -> pd.Series:
    """Worst drawdown within each trailing window."""
    wealth = (1 + r).cumprod()
    return wealth.rolling(window).apply(lambda w: (w / np.maximum.accumulate(w) - 1).min(), raw=True)


# --------------------------------------------------------------------------- #
# Summary
# --------------------------------------------------------------------------- #
def summary(r: pd.Series, rf: Optional[pd.Series] = None,
            bench: Optional[pd.Series] = None, var_level: float = 0.95) -> Dict[str, float]:
    var_, cvar_ = var_cvar(r, var_level)
    dd_tbl = drawdown_table(r, 1)
    out = {
        "Total return": total_return(r),
        "CAGR": cagr(r),
        "Volatility": ann_vol(r),
        "Downside deviation": downside_deviation(r, rf),
        "Sharpe": sharpe(r, rf),
        "Sortino": sortino(r, rf),
        "Max drawdown": max_drawdown(r),
        "Max DD length (days)": float(dd_tbl["total_days"].iloc[0]) if len(dd_tbl) else np.nan,
        "Calmar": calmar(r),
        f"Daily VaR {var_level:.0%}": var_,
        f"Daily CVaR {var_level:.0%}": cvar_,
        "Omega": omega_ratio(r),
        "Best month": monthly_returns(r).max(),
        "Worst month": monthly_returns(r).min(),
        "% positive months": float((monthly_returns(r) > 0).mean()),
    }
    out.update({k: v for k, v in tail_stats(r).items() if k in ("skew", "excess_kurtosis")})
    if bench is not None:
        ba = beta_alpha(r, bench, rf)
        cap = capture_ratios(r, bench)
        out.update({
            "Beta": ba["beta"], "Alpha (ann)": ba["alpha_ann"], "Alpha t-stat": ba["alpha_t"],
            "R-squared": ba["r_squared"], "Correlation": correlation(r, bench),
            "Tracking error": tracking_error(r, bench),
            "Information ratio": information_ratio(r, bench),
            "Up capture": cap["up_capture"], "Down capture": cap["down_capture"],
            "Monthly hit rate": cap["hit_rate"],
        })
    return out


PCT_KEYS = {"Total return", "CAGR", "Volatility", "Downside deviation", "Max drawdown",
            "Best month", "Worst month", "% positive months", "Alpha (ann)", "Tracking error",
            "Monthly hit rate", "Up capture", "Down capture"}


def format_summary(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().astype(object)
    for idx in out.index:
        for c in out.columns:
            v = df.loc[idx, c]
            if pd.isna(v):
                out.loc[idx, c] = ""
            elif idx in PCT_KEYS or str(idx).startswith("Daily VaR") or str(idx).startswith("Daily CVaR"):
                out.loc[idx, c] = f"{v:.2%}"
            elif "days" in str(idx):
                out.loc[idx, c] = f"{v:.0f}"
            else:
                out.loc[idx, c] = f"{v:.2f}"
    return out
