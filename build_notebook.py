"""Assemble notebooks/portfolio_risk_analytics.ipynb from src/riskdash."""
import nbformat as nbf
from pathlib import Path

ROOT = Path(__file__).parent
SRC = ROOT / "src" / "riskdash"
nb = nbf.v4.new_notebook()
nb.metadata = {"kernelspec": {"name": "python3", "display_name": "Python 3"},
               "language_info": {"name": "python"}, "colab": {"provenance": [], "toc_visible": True}}
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip("\n")))

md(r"""
# Portfolio Risk & Performance Analytics Dashboard

**Asset-allocation track · Project 1 of 6.** The reporting layer every later project plugs into.

Give it a list of tickers and weights and a benchmark. It produces the report an institutional
risk committee expects each month:

| Block | What it answers |
|---|---|
| Performance | CAGR, vol, Sharpe/Sortino/Calmar, VaR/CVaR, tail shape, monthly and annual tables |
| Benchmark-relative | beta, Jensen alpha with t-stat, tracking error, information ratio, up/down capture |
| **Risk decomposition** | which positions actually drive portfolio volatility (Euler MCTR/CCTR), by asset and asset class, static and over time |
| Diversification quality | correlation in calm vs stress days, does the hedge still hedge |
| Scenarios | today's weights through eight historical stress windows |
| What-if | ex-ante impact of a proposed weight change |
| Export | a single self-contained HTML dashboard |

The central chart is *weight vs share of risk*. In the default 9-ETF portfolio, 25% in Treasuries
carries 7% of the risk while 10% in gold carries 17%. Equal-looking allocations routinely hide
concentrated risk, and that gap is what this tool is for.

**Data:** Yahoo Finance (adjusted prices), FRED (local short rates). Market presets for US, India,
UK, euro area, Japan and UAE set the currency, risk-free rate, benchmark and stress windows in one
word; any Yahoo symbol works in any portfolio.
""")

md("## 1. Setup")
code(r"""
import importlib, subprocess, sys
def ensure(pkg, mod):
    try: importlib.import_module(mod); return
    except ImportError: pass
    for extra in ([], ["--break-system-packages"]):
        if subprocess.call([sys.executable, "-m", "pip", "install", "-q", pkg] + extra) == 0: return
    print(f"WARNING: could not install {pkg}")
for pkg, mod in [("yfinance", "yfinance"), ("scipy", "scipy")]:
    ensure(pkg, mod)
print("dependencies ready")
""")
code(r"""
import os, sys, warnings, logging, time
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

USE_DRIVE = False          # persist cache/outputs to Google Drive across sessions
if USE_DRIVE:
    from google.colab import drive  # type: ignore
    drive.mount("/content/drive"); os.chdir("/content/drive/MyDrive/portfolio-risk-analytics")

for d in ["src/riskdash", "data/cache", "outputs/figures", "outputs/tables", "tests"]:
    os.makedirs(d, exist_ok=True)
if os.path.abspath("src") not in sys.path:
    sys.path.insert(0, os.path.abspath("src"))
print("working directory:", os.getcwd())
""")

md(r"""
## 2. Package source

Written to `src/riskdash/` so the notebook is self-contained in Colab and the same code is
importable in a cloned repo.

```
portfolio-risk-analytics/
├── notebooks/portfolio_risk_analytics.ipynb
├── src/riskdash/
│   ├── config.py     portfolio, benchmark, asset classes, stress windows, settings
│   ├── data.py       Yahoo/FRED loaders, calendar alignment, FX conversion, quality gate
│   ├── metrics.py    performance, drawdown table, benchmark-relative, rolling statistics
│   ├── risk.py       simulation (drift vs rebalance), Euler decomposition, what-if, scenarios
│   ├── plots.py      figures
│   └── report.py     single-file HTML dashboard
├── tests/test_riskdash.py   12 offline tests
└── outputs/{figures,tables,dashboard.html}
```
""")
ORDER = ["__init__", "config", "data", "metrics", "risk", "plots", "report"]
BLURB = {
    "__init__": "Package version.",
    "config": "Portfolio and benchmark as `ticker -> weight` dicts (the benchmark can be a blend), an asset-class map for grouping, dashboard settings, and **market presets** (US, IN, UK, EU, JP, AE) that bundle currency, a FRED risk-free series, a default benchmark and local stress windows.",
    "data": "Downloads with retry and CSV caching, cross-exchange calendar alignment, repair of self-reversing bad prints (Yahoo occasionally publishes a −90%/+900% pair that would wreck every statistic), conversion to a base currency via Yahoo FX crosses, FRED risk-free series, a quality gate that reports everything it changed, and a synthetic fallback for offline runs.",
    "metrics": "Every statistic in the report, computed on daily returns with the daily T-bill series as the risk-free rate. Includes a drawdown episode table (peak / trough / recovery dates and durations), monthly and annual tables, CAPM beta and alpha with a t-stat, tracking error, information ratio and capture ratios.",
    "risk": "Portfolio simulation with either buy-and-hold drift or periodic rebalancing (with turnover accounting), Euler risk decomposition (marginal, component and percentage contributions, beta to portfolio, diversification ratio), rolling risk shares using drifted holdings, what-if analysis, historical stress tests and calm-vs-stress correlation.",
    "plots": "One function per figure; consistent style; nothing calls `plt.show()`.",
    "report": "Assembles KPI cards, embedded figures and formatted tables into one HTML file with no external dependencies.",
}
for name in ORDER:
    md(f"### `riskdash/{name}.py`\n\n{BLURB[name]}")
    code(f"%%writefile src/riskdash/{name}.py\n{(SRC / f'{name}.py').read_text()}")

code(r"""
import importlib, riskdash
for m in ["config", "data", "metrics", "risk", "plots", "report"]:
    importlib.reload(importlib.import_module(f"riskdash.{m}"))
from riskdash import data, metrics as met, risk, plots, report
from riskdash.config import PORTFOLIO, BENCHMARK, BENCHMARK_NAME, ASSET_CLASS, STRESS_SCENARIOS, DATA, DASH, DashboardConfig, market_preset, MARKETS
import numpy as np, pandas as pd, matplotlib.pyplot as plt
plots.set_style()
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 40); pd.set_option("display.precision", 4)
print("riskdash", riskdash.__version__)
""")

md(r"""
## 3. Define the portfolio

**Pick a market, then list your holdings.** The market preset sets the base currency, a local
risk-free rate from FRED, a default benchmark and the stress windows that matter locally:

| `MARKET` | Currency | Risk-free | Default benchmark | Ticker suffix |
|---|---|---|---|---|
| `US` | USD | 3-month T-bill | 60/40 SPY/AGG | none |
| `IN` | INR | RBI call rate | Nifty 50 (NIFTYBEES.NS) | `.NS` / `.BO` |
| `UK` | GBP | 3-month interbank | FTSE 100 (ISF.L) | `.L` |
| `EU` | EUR | 3-month Euribor | Euro Stoxx 50 (EXW1.DE) | `.DE` `.PA` `.MI` `.AS` |
| `JP` | JPY | 3-month interbank | Nikkei 225 (1321.T) | `.T` |
| `AE` | USD (peg) | US T-bill | MSCI UAE (UAE) | `.AE` |

Weights are normalised to one, so quantities × price work as well as percentages. Any holding
from any market can go in any portfolio; prices are converted to the preset's currency first.
""")
code(r"""
MARKET = "US"                                   # "US" | "IN" | "UK" | "EU" | "JP" | "AE"
preset = market_preset(MARKET)

# ---- your holdings: ticker -> weight (or current value; normalised automatically)
portfolio = dict(PORTFOLIO)                     # US multi-asset example
# portfolio = {"RELIANCE.NS": 25000, "HDFCBANK.NS": 18000, "TCS.NS": 15000, "GOLDBEES.NS": 12000}   # IN example (values in INR)

benchmark = dict(preset.benchmark)              # or your own, e.g. {"SPY": 1.0}
benchmark_name = preset.benchmark_name
asset_class = dict(ASSET_CLASS)                 # optional grouping; unmapped tickers show as "Other"
scenarios = dict(preset.scenarios)
data_cfg = preset.data_config(DATA)             # currency + risk-free + start date for this market
REBALANCE = DASH.rebalance                      # "ME" monthly, "QE" quarterly, None = buy-and-hold
USE_SYNTHETIC = False

if MARKET != "US" and portfolio == PORTFOLIO:
    print("NOTE: MARKET is", MARKET, "but the portfolio is still the US example; replace it with your holdings.")
print(f"{preset.name} | currency {data_cfg.base_currency} | risk-free: {preset.rf_note} | benchmark: {benchmark_name} | data from {data_cfg.start}")
pd.Series(portfolio, name="weight").pipe(lambda s: s / s.sum()).to_frame().T.style.format("{:.1%}")
""")

md("## 4. Data collection and quality checks")
code(r"""
t0 = time.time()
D = data.load_all(list(portfolio) + list(benchmark), data_cfg, use_synthetic=USE_SYNTHETIC)
returns, rf, prices = D["returns"], D["rf"], D["prices"]
print(f"loaded in {time.time()-t0:.1f}s | {prices.shape[0]} days x {prices.shape[1]} tickers | {prices.index[0].date()} -> {prices.index[-1].date()}")
print("issues:", D["issues"] or "none")
print(f"risk-free mean {rf.mean()*252:.2%} p.a., latest {rf.iloc[-1]*252:.2%}")
desc = pd.DataFrame({"ann_return": returns.mean()*252, "ann_vol": returns.std()*np.sqrt(252),
                     "max_drawdown": returns.apply(met.max_drawdown), "skew": returns.skew(), "kurt": returns.kurt()})
display(desc.style.format({"ann_return": "{:.1%}", "ann_vol": "{:.1%}", "max_drawdown": "{:.1%}", "skew": "{:.2f}", "kurt": "{:.1f}"}))
""")

md("## 5. Run the analytics")
code(r"""
t0 = time.time()
dash_cfg = DashboardConfig(**{**DASH.__dict__, "rebalance": REBALANCE})
R = risk.run_dashboard(returns, rf, portfolio, benchmark, benchmark_name, asset_class, dash_cfg, scenarios)
print(f"computed in {time.time()-t0:.1f}s | {R['portfolio'].index[0].date()} -> {R['portfolio'].index[-1].date()}")
summary = R["summary"]
summary.to_csv("outputs/tables/summary.csv")
display(met.format_summary(summary))
""")
md(r"""
Three columns on purpose. *Portfolio* is rebalanced back to target every period; *buy & hold*
lets weights drift, so the equity sleeve grows and the book quietly becomes riskier (compare the
betas and drawdowns). The difference between them is the value, or cost, of rebalancing
discipline.
""")

md("## 6. Growth, drawdowns and calendar returns")
code(r"""
fig = plots.plot_growth(R["portfolio"], R["benchmark"], R["buy_hold"], benchmark_name); plots.save(fig, "01_growth_drawdown"); plt.show()
display(R["drawdowns"].style.format({"depth": "{:.1%}", "days_to_recover": "{:.0f}"}))
R["drawdowns"].to_csv("outputs/tables/drawdowns.csv")
""")
code(r"""
fig = plots.plot_monthly_heatmap(R["monthly"]); plots.save(fig, "02_monthly_heatmap"); plt.show()
fig = plots.plot_annual(R["annual"]); plots.save(fig, "03_annual_returns"); plt.show()
R["monthly"].to_csv("outputs/tables/monthly_returns.csv")
""")

md("## 7. Rolling risk: volatility, beta, Sharpe")
code(r"""
fig = plots.plot_rolling(R["rolling"], benchmark_name, DASH.rolling_window, DASH.long_window); plots.save(fig, "04_rolling"); plt.show()
fig = plots.plot_return_distribution(R["portfolio"], R["benchmark"], DASH.var_level); plots.save(fig, "05_distribution"); plt.show()
""")

md(r"""
## 8. Risk decomposition: where the risk actually sits

Euler decomposition of portfolio volatility using the trailing 1-year covariance.
`risk_to_weight` above 1 means the position contributes more risk than capital.
""")
code(r"""
dec = R["decomposition"]
display(dec.style.format({"weight": "{:.1%}", "stand_alone_vol": "{:.1%}", "marginal_ctr": "{:.2%}", "component_ctr": "{:.2%}",
                          "pct_of_risk": "{:.1%}", "beta_to_portfolio": "{:.2f}", "risk_to_weight": "{:.2f}x"})
        .background_gradient(subset=["pct_of_risk"], cmap="Reds").background_gradient(subset=["risk_to_weight"], cmap="RdYlGn_r", vmin=0, vmax=2))
print(f"portfolio vol {dec.attrs['portfolio_vol']:.2%} | diversification ratio {dec.attrs['diversification_ratio']:.2f} | effective number of risk sources {dec.attrs['effective_n_risk']:.1f} (of {len(dec)})")
dec.to_csv("outputs/tables/risk_decomposition.csv")
fig = plots.plot_risk_vs_weight(dec); plots.save(fig, "06_risk_vs_weight"); plt.show()
""")
code(r"""
display(R["group_decomposition"].style.format({"weight": "{:.1%}", "component_ctr": "{:.2%}", "pct_of_risk": "{:.1%}", "risk_to_weight": "{:.2f}x"}))
fig = plots.plot_group_risk(R["group_decomposition"]); plots.save(fig, "07_group_risk"); plt.show()
fig = plots.plot_rolling_risk(R["rolling_risk"]); plots.save(fig, "08_rolling_risk"); plt.show()
""")

md(r"""
## 9. Does the diversification survive stress?

Average pairwise correlation and each asset's correlation to the benchmark on the benchmark's
worst 10% of days versus the rest. For a multi-asset book the pairwise average can *fall* in
stress because Treasuries rally when equities sell off; the per-asset view shows which sleeves
provided that offset and which (credit, REITs, EM) did not.
""")
code(r"""
display(R["regimes"]["summary"].style.format({"avg_pairwise_corr": "{:.2f}", "n_days": "{:.0f}"}))
display(R["regimes"]["to_benchmark"].style.format({"calm": "{:.2f}", "stress": "{:.2f}", "change": "{:+.2f}", "avg_ret_on_stress_days": "{:.2%}"})
        .background_gradient(subset=["stress"], cmap="RdBu_r", vmin=-1, vmax=1))
fig = plots.plot_regime_correlation(R["regimes"]["to_benchmark"], benchmark_name); plots.save(fig, "09_regime_correlation"); plt.show()
fig = plots.plot_correlation(R["regimes"]["full"], "Correlation, all days"); plots.save(fig, "10_correlation"); plt.show()
""")

md("## 10. Historical stress scenarios")
code(r"""
display(R["stress"].style.format({"portfolio": "{:.1%}", "benchmark": "{:.1%}", "worst_day": "{:.1%}", "worst_asset_return": "{:.1%}"})
        .background_gradient(subset=["portfolio"], cmap="RdYlGn", vmin=-0.3, vmax=0.1))
R["stress"].to_csv("outputs/tables/stress_scenarios.csv")
fig = plots.plot_stress(R["stress"], benchmark_name); plots.save(fig, "11_stress"); plt.show()
""")

md(r"""
## 11. What-if: a proposed reallocation

Absolute weight changes; the rest of the book is renormalised. Ex-ante vol and risk shares
before and after, using the same trailing covariance. Change the dict and re-run.
""")
code(r"""
CHANGES = {"GLD": -0.05, "IEF": +0.05}      # e.g. trim gold, add duration
CHANGES = {k: v for k, v in CHANGES.items() if k in R["weights"].index} or {R["decomposition"]["marginal_ctr"].idxmax(): -0.05, R["decomposition"]["marginal_ctr"].idxmin(): +0.05}
wi = risk.what_if(R["weights"], R["cov"], R["mu"], CHANGES)
print(f"change {CHANGES}: vol {wi.attrs['vol_before']:.2%} -> {wi.attrs['vol_after']:.2%}"
      + (f", historical-mean return {wi.attrs['ret_before']:.2%} -> {wi.attrs['ret_after']:.2%} (a noisy estimate; treat with caution)" if "ret_before" in wi.attrs else ""))
display(wi.style.format("{:.1%}"))
fig = plots.plot_what_if(wi); plots.save(fig, "12_what_if"); plt.show()
""")

md("## 12. Per-asset statistics")
code(r"""
display(met.format_summary(R["per_asset"]))
R["per_asset"].to_csv("outputs/tables/per_asset.csv")
""")

md(r"""
## 13. Export the dashboard

One HTML file with the KPI cards, every figure and the key tables embedded. Open it in a
browser, attach it to an e-mail, or drop it on a shared drive.
""")
code(r"""
figs = [
    ("Growth and drawdown", plots.plot_growth(R["portfolio"], R["benchmark"], R["buy_hold"], benchmark_name)),
    ("Weight vs risk contribution", plots.plot_risk_vs_weight(R["decomposition"])),
    ("Risk by asset class", plots.plot_group_risk(R["group_decomposition"])),
    ("Risk share over time", plots.plot_rolling_risk(R["rolling_risk"])),
    ("Rolling volatility, beta, Sharpe", plots.plot_rolling(R["rolling"], benchmark_name, DASH.rolling_window, DASH.long_window)),
    ("Monthly returns", plots.plot_monthly_heatmap(R["monthly"])),
    ("Correlation to benchmark: calm vs stress", plots.plot_regime_correlation(R["regimes"]["to_benchmark"], benchmark_name)),
    ("Stress scenarios", plots.plot_stress(R["stress"], benchmark_name)),
]
label = {"ME": "monthly rebalancing", "QE": "quarterly rebalancing", None: "buy and hold"}.get(REBALANCE, f"{REBALANCE} rebalancing")
path = report.build_report(R, figs, "Portfolio Risk & Performance Dashboard", "outputs/dashboard.html", label)
print("written:", path, f"({os.path.getsize(path)/1e6:.1f} MB)")
try:
    from google.colab import files  # type: ignore
    files.download(path)
except Exception:
    from IPython.display import HTML, display as _d
    _d(HTML(f'<a href="{path}" target="_blank">open dashboard.html</a>'))
""")

md(r"""
## 14. Reading the results

Write your own after each run; the pattern on the default portfolio has been:

1. **Capital allocation is not risk allocation.** The 25% Treasury sleeve carries under 10% of
   the risk; gold at 10% carries close to 20%; EM equity at 5% roughly doubles its weight in risk
   terms. The book is ~75% equity-like in risk while looking 55% equity by weight.
2. **Rebalancing changes the risk profile more than the return.** Buy-and-hold drifted to a
   higher beta and deeper drawdowns for a similar Sharpe; the value of rebalancing shows up in
   risk control, not in alpha.
3. **The hedge mostly held; credit did not.** Against a 60/40 benchmark (which already holds
   bonds) Treasury correlation goes mildly negative on stress days, while investment-grade credit
   correlation *rises* (roughly 0.3 → 0.45): credit behaves like equity exactly when you need it
   not to. 2022 is the exception that proves the rule: when rates were the shock, the Treasury
   sleeve deepened the drawdown instead of cushioning it.
4. **Alpha versus a 60/40 is statistically zero.** The t-stat on Jensen's alpha is well below 2;
   what differs is the shape of risk, not the level of return.

## 15. Potential upgrades

| Upgrade | Why |
|---|---|
| Factor-based risk decomposition (equity, rates, credit, FX, commodity betas) | Project 2 in this track; explains *why* two assets co-move |
| Ex-ante VaR/CVaR via Monte Carlo with fat-tailed marginals and a t-copula | Historical VaR is bounded by the sample; the 2008 tail is not in a 2010-start dataset |
| Liquidity-adjusted risk and position-size limits | Turns the dashboard from descriptive into a limits framework |
| Attribution: allocation vs selection vs interaction (Brinson) against the benchmark | Standard institutional monthly reporting |
| Scheduled run (GitHub Actions cron) with e-mailed HTML and threshold alerts | The operating version of this notebook |
| Interactive front end (Plotly Dash / Streamlit) with weight sliders driving the what-if | For portfolio managers who will not open a notebook |
""")

nb.cells = cells
out = ROOT / "notebooks" / "portfolio_risk_analytics.ipynb"
nbf.write(nb, out)
print("wrote", out, "with", len(cells), "cells")
