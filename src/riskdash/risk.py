"""
Portfolio construction, risk decomposition and scenario analysis.

Risk decomposition (Euler)
--------------------------
For portfolio volatility sigma_p = sqrt(w' Sigma w):

    MCTR_i = (Sigma w)_i / sigma_p          marginal contribution to risk
    CCTR_i = w_i * MCTR_i                    component contribution (sums to sigma_p)
    PCTR_i = CCTR_i / sigma_p                percentage contribution (sums to 1)

A 5% weight can be 20% of the risk if the asset is volatile and correlated
with the rest of the book; that gap between weight and risk share is the
central message of a risk dashboard.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from . import metrics as met
from .config import DASH, STRESS_SCENARIOS, DashboardConfig

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Weights
# --------------------------------------------------------------------------- #
def normalise_weights(w: Dict[str, float]) -> pd.Series:
    s = pd.Series(w, dtype=float)
    if (s < 0).any():
        raise ValueError("Negative weights are not supported in this dashboard")
    if s.sum() <= 0:
        raise ValueError("Weights must sum to a positive number")
    if abs(s.sum() - 1.0) > 1e-6:
        logger.info("Weights sum to %.4f; normalising to 1", s.sum())
    return s / s.sum()


# --------------------------------------------------------------------------- #
# Simulation
# --------------------------------------------------------------------------- #
def _segment(w0: np.ndarray, seg: pd.DataFrame) -> Tuple[pd.Series, pd.DataFrame]:
    growth = (1.0 + seg.values).cumprod(axis=0)
    value = growth * w0
    total = value.sum(axis=1)
    ret = np.empty(len(seg))
    ret[0] = total[0] - 1.0
    ret[1:] = total[1:] / total[:-1] - 1.0
    return (pd.Series(ret, index=seg.index),
            pd.DataFrame(value / total[:, None], index=seg.index, columns=seg.columns))


def simulate(returns: pd.DataFrame, weights: pd.Series,
             rebalance: Optional[str] = "ME") -> Tuple[pd.Series, pd.DataFrame, pd.Series]:
    """
    Portfolio daily returns and daily drifted holdings.

    rebalance : pandas offset alias ("ME", "QE", "YE") or None for buy-and-hold.
    Returns (portfolio_returns, holdings, traded) where `traded` is the sum of
    absolute weight changes on each rebalance date (turnover accounting).
    """
    weights = weights.reindex(returns.columns).fillna(0.0)
    w = weights.values
    if rebalance is None:
        ret, hold = _segment(w, returns)
        traded = pd.Series(0.0, index=returns.index)
        return ret, hold, traded

    period_ends = returns.resample(rebalance).last().index
    # map period ends to actual trading days
    locs = sorted(set(returns.index.searchsorted(d, side="right") - 1 for d in period_ends))
    locs = [l for l in locs if 0 <= l < len(returns) - 1]
    bounds = [0] + [l + 1 for l in locs] + [len(returns)]
    rets, holds, traded = [], [], pd.Series(0.0, index=returns.index)
    prev_hold = None
    for a, b in zip(bounds[:-1], bounds[1:]):
        if b <= a:
            continue
        seg = returns.iloc[a:b]
        if prev_hold is not None:
            traded.iloc[a] = float(np.abs(w - prev_hold).sum())
        r, h = _segment(w, seg)
        rets.append(r); holds.append(h)
        prev_hold = h.iloc[-1].values
    return pd.concat(rets), pd.concat(holds), traded


def build_series(returns: pd.DataFrame, spec: Dict[str, float],
                 rebalance: Optional[str], name: str) -> Tuple[pd.Series, pd.DataFrame, pd.Series]:
    w = normalise_weights(spec)
    missing = [t for t in w.index if t not in returns.columns]
    if missing:
        raise KeyError(f"Tickers missing from returns: {missing}")
    port, hold, traded = simulate(returns[w.index], w, rebalance)
    port.name = name
    return port, hold, traded


# --------------------------------------------------------------------------- #
# Risk decomposition
# --------------------------------------------------------------------------- #
def risk_decomposition(weights: pd.Series, cov: pd.DataFrame) -> pd.DataFrame:
    """
    Per-asset table: weight, stand-alone vol, marginal / component / percent
    contribution to portfolio vol, beta to the portfolio, and the ratio of
    risk share to weight share (>1 means the asset punches above its weight).
    """
    assets = list(cov.index)
    w = weights.reindex(assets).fillna(0.0).values
    S = cov.values
    port_var = float(w @ S @ w)
    port_vol = np.sqrt(port_var)
    mctr = S @ w / port_vol
    cctr = w * mctr
    pctr = cctr / port_vol
    beta = (S @ w) / port_var
    tbl = pd.DataFrame({
        "weight": w,
        "stand_alone_vol": np.sqrt(np.diag(S)),
        "marginal_ctr": mctr,
        "component_ctr": cctr,
        "pct_of_risk": pctr,
        "beta_to_portfolio": beta,
        "risk_to_weight": np.where(w > 0, pctr / np.where(w > 0, w, 1), np.nan),
    }, index=assets)
    tbl.attrs["portfolio_vol"] = port_vol
    tbl.attrs["diversification_ratio"] = float((w * np.sqrt(np.diag(S))).sum() / port_vol)
    tbl.attrs["effective_n_risk"] = float(1.0 / np.sum(pctr ** 2))
    return tbl


def group_decomposition(decomp: pd.DataFrame, mapping: Dict[str, str]) -> pd.DataFrame:
    grp = pd.Series({t: mapping.get(t, "Other") for t in decomp.index})
    out = decomp.groupby(grp)[["weight", "component_ctr", "pct_of_risk"]].sum()
    out["risk_to_weight"] = out["pct_of_risk"] / out["weight"]
    return out.sort_values("pct_of_risk", ascending=False)


def rolling_risk_contribution(holdings: pd.DataFrame, returns: pd.DataFrame,
                              window: int = 252, freq: str = "ME") -> pd.DataFrame:
    """Percent risk contribution using drifted holdings and a trailing covariance, at period ends."""
    dates = holdings.resample(freq).last().index
    rows = {}
    for d in dates:
        loc = returns.index.searchsorted(d, side="right") - 1
        if loc < window:
            continue
        date = returns.index[loc]
        cov = returns.iloc[loc - window + 1: loc + 1].cov() * met.TRADING_DAYS
        w = holdings.loc[:date].iloc[-1]
        rows[date] = risk_decomposition(w, cov)["pct_of_risk"]
    return pd.DataFrame(rows).T


# --------------------------------------------------------------------------- #
# What-if
# --------------------------------------------------------------------------- #
def what_if(weights: pd.Series, cov: pd.DataFrame, mu: Optional[pd.Series],
            changes: Dict[str, float]) -> pd.DataFrame:
    """
    Apply weight changes (absolute, e.g. {"TLT": -0.05, "GLD": +0.05}), renormalise,
    and compare ex-ante vol, risk shares and (if mu given) expected return.
    """
    new = weights.copy()
    for t, dw in changes.items():
        if t not in new.index:
            raise KeyError(f"{t} not in portfolio")
        new[t] = max(new[t] + dw, 0.0)
    new = new / new.sum()
    before, after = risk_decomposition(weights, cov), risk_decomposition(new, cov)
    tbl = pd.DataFrame({
        "weight_before": before["weight"], "weight_after": after["weight"],
        "risk_share_before": before["pct_of_risk"], "risk_share_after": after["pct_of_risk"],
    })
    tbl.attrs["vol_before"] = before.attrs["portfolio_vol"]
    tbl.attrs["vol_after"] = after.attrs["portfolio_vol"]
    if mu is not None:
        tbl.attrs["ret_before"] = float(mu.reindex(weights.index).fillna(0) @ weights)
        tbl.attrs["ret_after"] = float(mu.reindex(new.index).fillna(0) @ new)
    return tbl


# --------------------------------------------------------------------------- #
# Scenarios and regimes
# --------------------------------------------------------------------------- #
def stress_test(returns: pd.DataFrame, weights: pd.Series, bench: pd.Series,
                scenarios: Dict[str, tuple] = STRESS_SCENARIOS) -> pd.DataFrame:
    """
    Historical scenario P&L: apply *today's* weights (buy-and-hold within the
    window) to the actual asset returns over each stress window. Also reports
    the benchmark and the worst single day inside the window.
    """
    rows = []
    for name, (a, b) in scenarios.items():
        seg = returns.loc[a:b, weights.index]
        if len(seg) < 2:
            continue
        port, _ = _segment(weights.values, seg)
        bseg = bench.loc[a:b]
        rows.append(dict(scenario=name, start=seg.index[0].date(), end=seg.index[-1].date(),
                         days=len(seg), portfolio=met.total_return(port), benchmark=met.total_return(bseg),
                         worst_day=float(port.min()),
                         worst_asset=seg.apply(met.total_return).idxmin(),
                         worst_asset_return=float(seg.apply(met.total_return).min())))
    return pd.DataFrame(rows).set_index("scenario")


def correlation_regimes(returns: pd.DataFrame, bench: pd.Series,
                        quantile: float = 0.10) -> Dict[str, pd.DataFrame]:
    """
    Average pairwise correlation in the worst `quantile` of benchmark days versus
    all days. Correlations rising in the tail is the diversification failure
    every multi-asset investor should be looking for.
    """
    b = bench.reindex(returns.index)
    cut = b.quantile(quantile)
    stress = returns[b <= cut]
    calm = returns[b > cut]
    def _avg(c):
        m = c.values[np.triu_indices(len(c), 1)]
        return float(np.nanmean(m))
    full_c, stress_c, calm_c = returns.corr(), stress.corr(), calm.corr()
    summary = pd.DataFrame({
        "avg_pairwise_corr": [_avg(full_c), _avg(calm_c), _avg(stress_c)],
        "n_days": [len(returns), len(calm), len(stress)],
    }, index=["All days", f"Benchmark above {quantile:.0%} tail", f"Benchmark worst {quantile:.0%} days"])
    # Per-asset correlation to the benchmark by regime: a hedge that only
    # works on calm days is not a hedge.
    to_bench = pd.DataFrame({
        "calm": calm.corrwith(b[b > cut]),
        "stress": stress.corrwith(b[b <= cut]),
    })
    to_bench["change"] = to_bench["stress"] - to_bench["calm"]
    to_bench["avg_ret_on_stress_days"] = stress.mean()
    return dict(summary=summary, full=full_c, stress=stress_c, calm=calm_c,
                to_benchmark=to_bench.sort_values("stress"))


def exposure_by_group(weights: pd.Series, mapping: Dict[str, str]) -> pd.Series:
    grp = pd.Series({t: mapping.get(t, "Other") for t in weights.index})
    return weights.groupby(grp).sum().sort_values(ascending=False)


# --------------------------------------------------------------------------- #
# End-to-end
# --------------------------------------------------------------------------- #
def run_dashboard(returns: pd.DataFrame, rf: pd.Series, portfolio: Dict[str, float],
                  benchmark: Dict[str, float], benchmark_name: str,
                  asset_class: Dict[str, str], cfg: DashboardConfig = DASH,
                  scenarios: Dict[str, tuple] = STRESS_SCENARIOS) -> Dict[str, object]:
    """Compute everything the dashboard shows and return it in one dict."""
    w = normalise_weights(portfolio)
    port, hold, traded = build_series(returns, portfolio, cfg.rebalance, "Portfolio")
    bench, _, _ = build_series(returns, benchmark, "ME", benchmark_name)
    bh, _, _ = build_series(returns, portfolio, None, "Portfolio (buy & hold)")
    idx = port.index.intersection(bench.index)
    port, bench, bh = port.loc[idx], bench.loc[idx], bh.loc[idx]

    cov_window = returns[w.index].iloc[-cfg.cov_window:]
    cov = cov_window.cov() * met.TRADING_DAYS
    mu = returns[w.index].mean() * met.TRADING_DAYS
    decomp = risk_decomposition(w, cov)

    summ = pd.DataFrame({
        "Portfolio": met.summary(port, rf, bench, cfg.var_level),
        benchmark_name: met.summary(bench, rf, None, cfg.var_level),
        "Portfolio (buy & hold)": met.summary(bh, rf, bench, cfg.var_level),
    })
    per_asset = pd.DataFrame({t: met.summary(returns[t].loc[idx], rf, bench, cfg.var_level) for t in w.index})

    rolling = pd.DataFrame({
        "vol": met.rolling_vol(port, cfg.rolling_window),
        "bench_vol": met.rolling_vol(bench, cfg.rolling_window),
        "beta": met.rolling_beta(port, bench, cfg.rolling_window),
        "corr": met.rolling_correlation(port, bench, cfg.rolling_window),
        "sharpe_3y": met.rolling_sharpe(port, rf, cfg.long_window),
        "bench_sharpe_3y": met.rolling_sharpe(bench, rf, cfg.long_window),
        "drawdown": met.drawdown_series(port),
        "bench_drawdown": met.drawdown_series(bench),
    })

    return dict(
        weights=w, portfolio=port, benchmark=bench, buy_hold=bh, holdings=hold, traded=traded,
        cov=cov, mu=mu, decomposition=decomp,
        group_decomposition=group_decomposition(decomp, asset_class),
        exposure=exposure_by_group(w, asset_class),
        summary=summ, per_asset=per_asset, rolling=rolling,
        drawdowns=met.drawdown_table(port, cfg.top_drawdowns),
        bench_drawdowns=met.drawdown_table(bench, cfg.top_drawdowns),
        monthly=met.monthly_table(port), annual=pd.DataFrame({"Portfolio": met.annual_returns(port),
                                                             benchmark_name: met.annual_returns(bench)}),
        stress=stress_test(returns, w, bench, scenarios),
        regimes=correlation_regimes(returns[w.index].loc[idx], bench),
        rolling_risk=rolling_risk_contribution(hold, returns[w.index], cfg.cov_window),
        benchmark_name=benchmark_name,
    )
