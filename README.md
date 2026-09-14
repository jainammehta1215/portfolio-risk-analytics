# Portfolio Risk & Performance Analytics Dashboard

**Asset-allocation track · Project 1 of 6** · reporting layer for the rest of the series · see
[Project 3, Portfolio Optimisation](https://github.com/jainammehta1215/portfolio-optimisation) for
the allocation side

Give it tickers, weights and a benchmark. It produces the monthly report an institutional risk
committee expects, with the one chart most retail tools never show: how much of the portfolio's
risk each position actually carries.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jainammehta1215/portfolio-risk-analytics/blob/main/notebooks/portfolio_risk_analytics.ipynb)
[![tests](https://github.com/jainammehta1215/portfolio-risk-analytics/actions/workflows/tests.yml/badge.svg)](https://github.com/jainammehta1215/portfolio-risk-analytics/actions)

---

## 1. Project overview

A capital allocation is not a risk allocation. In the default nine-ETF portfolio, 25% in
Treasuries carries about 7% of the risk while 10% in gold carries 17% and 5% in emerging-market
equity carries 10%. The book looks 55% equity by weight and is roughly 75% equity-like by risk.

This project computes and visualises that gap, alongside the full performance and
benchmark-relative statistics, drawdown episodes, calendar tables, rolling risk, calm-versus-stress
correlation, historical scenario losses and an ex-ante what-if for proposed trades. Output is a
notebook and a single self-contained HTML dashboard.

## 2. Real-world finance use case

* **Monthly client and committee reporting** at asset managers and multi-family offices: the
  performance table, drawdown history and risk-contribution chart are the standard pack.
* **Risk budgeting:** the Euler decomposition here is the same arithmetic that the risk-parity
  optimiser in [Project 3](https://github.com/jainammehta1215/portfolio-optimisation) inverts, and
  that Project 4 builds on.
* **Pre-trade checks:** the what-if block answers "if I move 5% from gold to duration, what
  happens to ex-ante vol and to who owns the risk?" before the order goes in.
* **Hedge validation:** correlation on the benchmark's worst days tells you whether a sleeve
  bought for diversification actually diversifies when it matters.

## 3. System architecture

```
config.py ──► data.py ──► returns, rf ──► risk.run_dashboard() ──► R (dict of results)
   │            │                              │                        │
   │       Yahoo / FRED                 simulate (drift or rebalance)   ├─► notebook tables & figures
   │       calendar alignment           Euler decomposition             └─► report.build_report() ─► dashboard.html
   │       FX to base currency          rolling risk shares
   └── portfolio, benchmark,            what-if, stress, regimes
       asset classes, windows           metrics.summary()
```

Design decisions: one `run_dashboard()` call returns every table so the notebook and the HTML
report render from the same object; gross returns and traded notional are tracked separately;
buy-and-hold and rebalanced versions are always computed together because their difference is the
cost or value of rebalancing discipline.

## 4. Required APIs and data sources

| Source | What | Access |
|---|---|---|
| Yahoo Finance (`yfinance`) | Adjusted daily closes for any listed ticker; FX crosses for currency conversion | Free, no key |
| FRED `fredgraph.csv` | `DGS3MO` 3-month T-bill as the risk-free rate | Free, no key |
| Synthetic generator | Deterministic GBM for offline runs and tests | built in |

## 5. Required Python libraries

`numpy pandas scipy matplotlib yfinance requests` · dev: `pytest nbformat nbconvert ipykernel`

## 6. Folder / file structure

```
portfolio-risk-analytics/
├── README.md · requirements.txt · pyproject.toml · build_notebook.py
├── notebooks/portfolio_risk_analytics.ipynb      self-contained; writes src/ itself in Colab
├── src/riskdash/
│   ├── config.py      portfolio, benchmark, asset classes, stress windows, settings
│   ├── data.py        loaders, calendar alignment, FX conversion, quality gate
│   ├── metrics.py     performance, drawdown table, benchmark-relative, rolling
│   ├── risk.py        simulation, Euler decomposition, what-if, scenarios, regimes
│   ├── plots.py       figures
│   └── report.py      HTML dashboard
├── tests/test_riskdash.py                        12 offline tests
├── .github/workflows/tests.yml                   CI
└── outputs/{figures,tables,dashboard.html}
```

## 7. Step-by-step build guide

1. Config: portfolio and benchmark as `ticker -> weight` dicts, an asset-class map, stress windows.
2. Data layer with a quality gate; convert everything to a base currency before computing returns.
3. Metrics module, each statistic its own tested function; drawdown *episodes*, not just the minimum.
4. Simulation with drift between rebalances and turnover measured against drifted holdings.
5. Euler decomposition; verify percentage contributions sum to one and component contributions sum to portfolio vol.
6. Rolling risk shares using drifted holdings and a trailing covariance.
7. Regime correlation, historical stress windows, what-if.
8. Plots, then the HTML report assembled from the same result object.
9. Tests, notebook assembled from the modules, executed top to bottom before publishing.

## 8. Data collection pipeline

`data.load_all(tickers)` → `yf.download(auto_adjust=True)` → tz-naive index → CSV cache → drop
days when fewer than half the tickers traded (cross-exchange holidays) → forward-fill isolated gaps
(≤ 5 days) → drop tickers with < 98% coverage (with a message giving their first price date) →
convert to base currency via Yahoo FX crosses (pence/cents scaled first) → simple daily returns →
FRED yield to a daily rate aligned to the trading calendar. All network calls retry with back-off;
failures log loudly rather than defaulting silently.

## 9. Data cleaning and feature engineering

Self-reversing bad prints (a −90%/+900% pair within a few days) are detected and repaired before anything else; Yahoo publishes these occasionally and one of them wrecks every statistic downstream. Simple returns (they aggregate linearly across positions). Portfolio series built two ways
(rebalanced and buy-and-hold). Trailing 1-year annualised covariance for decomposition. Monthly
aggregation for capture ratios and the hit rate. Benchmark worst-decile days as the stress regime.
Absolute return over fixed date windows for scenarios, using today's weights.

## 10. Core models / algorithms

| Component | Formula / method |
|---|---|
| Portfolio return | segment-wise buy-and-hold: value_t = w₀ ∏(1+r), return = Δ total value; reset at each rebalance |
| Euler decomposition | MCTR = Σw/σₚ; CCTR = w∘MCTR; PCTR = CCTR/σₚ; β_to_portfolio = Σw/σₚ² |
| Diversification ratio | Σ wᵢσᵢ / σₚ |
| Effective risk sources | 1 / Σ PCTRᵢ² |
| Beta, alpha | OLS of excess returns; alpha annualised; t-stat on the intercept |
| Tracking error, IR | std and mean/std of active daily returns, annualised |
| Capture ratios | geometric mean of portfolio vs benchmark in up / down months |
| Drawdown episodes | peak → trough → recovery scan on compounded wealth |
| VaR / CVaR | historical, daily, 95% |
| Regimes | correlation matrices on benchmark ≤ 10th percentile days vs the rest; per-asset correlation to benchmark by regime |
| Stress | today's weights held through each window; worst asset and worst day inside it |

## 11. Visualisations and dashboard components

Growth of $1 with drawdown panel · monthly return heat-map with year totals · calendar-year bars ·
rolling vol / beta & correlation / 3-year Sharpe · daily return distribution with VaR/CVaR lines ·
weight vs risk-share bars with risk-to-weight multipliers · capital vs risk donuts by asset class ·
risk-share-over-time stack · calm vs stress correlation bars · full correlation heat-map · stress
scenario bars · what-if before/after bars. All embedded in `outputs/dashboard.html` with KPI cards.

## 12. Performance metrics

Total return, CAGR, volatility, downside deviation, Sharpe, Sortino, max drawdown and its length,
Calmar, daily VaR and CVaR 95%, Omega, best/worst month, % positive months, skew, excess kurtosis;
versus benchmark: beta, alpha and its t-stat, R², correlation, tracking error, information ratio,
up/down capture, monthly hit rate.

Default portfolio, 2010 → present, monthly rebalanced vs 60/40 (SPY/AGG): CAGR 9.6% vs 9.6%,
vol 10.5% vs 10.4%, Sharpe 0.78 vs 0.79, max drawdown −24.6% vs −21.6%, beta 0.96, alpha
+0.3% (t = 0.4), tracking error 3.1%. Buy-and-hold drifted to beta 1.17 and a −27% drawdown.

## 13. Final deliverables

`notebooks/portfolio_risk_analytics.ipynb` (executed, ~2 min in Colab) · `src/riskdash/` package ·
12 tests with CI · 12 figures · tables (summary, decomposition, drawdowns, scenarios, monthly,
per-asset) · `outputs/dashboard.html`.

## 14. Potential upgrades

Factor-based risk decomposition (Project 2) · Monte Carlo ex-ante VaR/CVaR with a t-copula ·
Brinson attribution · scheduled runs with e-mailed report and threshold alerts · Streamlit or Dash
front end with weight sliders · liquidity-adjusted risk and a limits framework.

## 15. Using your own portfolio

Pick a market preset, then list holdings as weights or current values:

```python
MARKET = "IN"
portfolio = {"RELIANCE.NS": 25000, "HDFCBANK.NS": 18000, "TCS.NS": 15000, "GOLDBEES.NS": 12000}
```

| `MARKET` | Currency | Risk-free (FRED) | Default benchmark |
|---|---|---|---|
| `US` | USD | 3-month T-bill | 60/40 SPY/AGG |
| `IN` | INR | RBI call money rate | Nifty 50 (NIFTYBEES.NS) |
| `UK` | GBP | 3-month interbank | FTSE 100 (ISF.L) |
| `EU` | EUR | 3-month Euribor | Euro Stoxx 50 (EXW1.DE) |
| `JP` | JPY | 3-month interbank | Nikkei 225 (1321.T) |
| `AE` | USD (AED peg) | US T-bill | MSCI UAE (UAE) |

Each preset also carries its own stress windows (for India: taper tantrum, demonetisation, IL&FS,
election result day 2024, the 2024–25 PSU correction, and so on). Holdings from any market can be
mixed; prices are converted to the preset currency. Tickers listed after the start date are dropped
with a message; set `DataConfig.start` later to keep them. `REBALANCE = None` gives buy-and-hold.

## Running it

```bash
pip install -r requirements.txt
PYTHONPATH=src pytest -q tests/            # 12 passed
python build_notebook.py                   # regenerate notebook from src/
```
Or open the notebook in Colab and Run all.

## Asset-allocation track

Six projects that build one capability each and share a reporting layer. Completed ones are linked.

| # | Project | What it adds |
|---|---|---|
| 1 | **Portfolio Risk & Performance Analytics Dashboard** (this repo) | Where the risk sits: Euler decomposition, benchmark-relative statistics, stress scenarios, HTML dashboard |
| 2 | [Multi-Factor Exposure Analyser (Fama-French)](https://github.com/jainammehta1215/factor-exposure-analyser) | Why assets co-move: factor betas, alpha after factor adjustment, style drift |
| 3 | [Portfolio Optimisation: Mean-Variance, Black-Litterman and Robust Methods](https://github.com/jainammehta1215/portfolio-optimisation) | What weights to hold: Markowitz and its fixes, tested out of sample net of costs |
| 4 | [Risk Parity and Hierarchical Risk Parity](https://github.com/jainammehta1215/risk-parity-hrp) | Allocating by risk instead of capital; clustering instead of matrix inversion |
| 5 | [Macro Nowcasting and Recession Probability](https://github.com/jainammehta1215/macro-nowcasting) | The regime the allocation lives in: yield-curve probit, dynamic factor model, real-time vintages |
| 6 | Yield Curve Construction and Fixed-Income Immunisation | The rates side: bootstrapping, Nelson-Siegel, key-rate durations, liability matching |

A master repository will consolidate all six with a shared core once the track is complete.
