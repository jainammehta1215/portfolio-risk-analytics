"""Offline tests on synthetic data. Run: PYTHONPATH=src pytest -q tests/"""
import numpy as np
import pandas as pd
import pytest

from riskdash import data, metrics as met, risk

TICKERS = ["A", "B", "C", "D", "E"]


@pytest.fixture(scope="module")
def rets():
    px = data.synthetic_prices(TICKERS, n_days=1300, seed=11)
    return data.compute_returns(px)


# --------------------------------------------------------------------------- #
# Simulation
# --------------------------------------------------------------------------- #
def test_buy_and_hold_matches_manual(rets):
    w = risk.normalise_weights({"A": 0.5, "B": 0.5})
    port, hold, traded = risk.simulate(rets[["A", "B"]], w, rebalance=None)
    wealth = 0.5 * (1 + rets["A"]).cumprod() + 0.5 * (1 + rets["B"]).cumprod()
    manual = wealth.pct_change().fillna(wealth.iloc[0] - 1)
    assert np.allclose(port.values, manual.values, atol=1e-12)
    assert np.allclose(hold.sum(axis=1), 1.0)
    assert traded.sum() == 0


def test_monthly_rebalance_resets_weights(rets):
    w = risk.normalise_weights({t: 1 / 5 for t in TICKERS})
    port, hold, traded = risk.simulate(rets, w, rebalance="ME")
    assert np.allclose(hold.sum(axis=1), 1.0)
    # on the first day of each month holdings equal target (before that day's drift, ~within a day of drift)
    firsts = hold.groupby([hold.index.year, hold.index.month]).head(1)
    assert (firsts.sub(0.2).abs().max(axis=1) < 0.03).all()
    assert (traded[traded > 0] > 0).all() and traded[traded > 0].max() < 0.5


def test_weights_normalise_and_reject_negative():
    assert abs(risk.normalise_weights({"A": 2, "B": 2}).sum() - 1) < 1e-12
    with pytest.raises(ValueError):
        risk.normalise_weights({"A": 1.2, "B": -0.2})


# --------------------------------------------------------------------------- #
# Risk decomposition
# --------------------------------------------------------------------------- #
def test_euler_decomposition_sums(rets):
    w = risk.normalise_weights({t: x for t, x in zip(TICKERS, [0.4, 0.3, 0.15, 0.1, 0.05])})
    cov = rets.cov() * 252
    d = risk.risk_decomposition(w, cov)
    assert abs(d["pct_of_risk"].sum() - 1) < 1e-10
    assert abs(d["component_ctr"].sum() - d.attrs["portfolio_vol"]) < 1e-10
    assert abs((d["weight"] * d["beta_to_portfolio"]).sum() - 1) < 1e-10
    assert d.attrs["diversification_ratio"] >= 1.0


def test_single_asset_has_all_risk(rets):
    w = risk.normalise_weights({"A": 1.0, "B": 0.0})
    d = risk.risk_decomposition(w, rets[["A", "B"]].cov() * 252)
    assert abs(d.loc["A", "pct_of_risk"] - 1) < 1e-12 and abs(d.loc["B", "pct_of_risk"]) < 1e-12


def test_what_if_renormalises_and_moves_vol(rets):
    w = risk.normalise_weights({t: 0.2 for t in TICKERS})
    cov = rets.cov() * 252
    # Moving weight from the highest to the lowest *marginal* contributor must
    # reduce vol (first-order). Stand-alone vol is not the right criterion:
    # a low-vol but high-correlation asset can add more risk than it removes.
    d = risk.risk_decomposition(w, cov)
    hi, lo = d["marginal_ctr"].idxmax(), d["marginal_ctr"].idxmin()
    tbl = risk.what_if(w, cov, None, {hi: -0.05, lo: +0.05})
    assert abs(tbl["weight_after"].sum() - 1) < 1e-12
    assert tbl.attrs["vol_after"] < tbl.attrs["vol_before"]


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def test_drawdown_table_on_constructed_series():
    idx = pd.bdate_range("2020-01-01", periods=10)
    r = pd.Series([0.1, -0.2, -0.1, 0.5, 0.0, -0.05, 0.1, 0.0, 0.0, 0.0], index=idx)
    tbl = met.drawdown_table(r, top=3)
    assert len(tbl) == 2
    assert abs(tbl.loc[0, "depth"] - (0.8 * 0.9 - 1)) < 1e-12        # -28% episode
    assert tbl.loc[0, "peak"] == idx[0] and tbl.loc[0, "trough"] == idx[2] and tbl.loc[0, "recovery"] == idx[3]
    assert pd.isna(tbl.loc[1, "recovery"]) is False or tbl.loc[1, "depth"] == pytest.approx(-0.05)


def test_tracking_error_zero_for_identical_series(rets):
    assert met.tracking_error(rets["A"], rets["A"]) == 0.0
    assert met.beta_alpha(rets["A"], rets["A"])["beta"] == pytest.approx(1.0)


def test_capture_ratios_leveraged_series(rets):
    b = rets["A"]
    lev = 1.5 * b
    cap = met.capture_ratios(lev, b)
    assert cap["up_capture"] > 1.2 and cap["down_capture"] > 1.2


def test_monthly_table_year_totals(rets):
    r = rets["B"]
    tbl = met.monthly_table(r)
    yr = tbl.index[1]
    expected = float((1 + r[r.index.year == yr]).prod() - 1)
    assert abs(tbl.loc[yr, "Year"] - expected) < 1e-10


def test_var_cvar_ordering(rets):
    v, cv = met.var_cvar(rets["C"], 0.95)
    assert cv >= v > 0


# --------------------------------------------------------------------------- #
# End-to-end on synthetic data
# --------------------------------------------------------------------------- #
def test_run_dashboard_end_to_end(rets):
    rf = pd.Series(0.0001, index=rets.index)
    R = risk.run_dashboard(rets, rf, {"A": 0.4, "B": 0.3, "C": 0.3}, {"D": 0.6, "E": 0.4}, "Bench",
                           {"A": "X", "B": "X", "C": "Y", "D": "Z", "E": "Z"})
    assert set(R["summary"].columns) == {"Portfolio", "Bench", "Portfolio (buy & hold)"}
    assert abs(R["exposure"].sum() - 1) < 1e-12
    assert len(R["drawdowns"]) >= 1
    assert R["rolling_risk"].shape[1] == 3
