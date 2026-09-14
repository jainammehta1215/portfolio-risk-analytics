"""
Dashboard figures. Each function returns a matplotlib Figure; nothing here
calls plt.show(), so the notebook and the HTML report control display.
"""
from __future__ import annotations

import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, PercentFormatter

from . import metrics as met

PALETTE = ["#1f4e79", "#c0392b", "#e67e22", "#27ae60", "#8e44ad",
           "#16a085", "#7f8c8d", "#d4ac0d", "#2c3e50", "#e84393"]
PORT_COLOR, BENCH_COLOR, BH_COLOR = "#1f4e79", "#7f8c8d", "#e67e22"
_pct = PercentFormatter(1.0, decimals=0)
_pct1 = PercentFormatter(1.0, decimals=1)


def set_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 200, "figure.facecolor": "white",
        "axes.grid": True, "grid.alpha": 0.3, "axes.spines.top": False, "axes.spines.right": False,
        "axes.titleweight": "bold", "axes.titlesize": 12, "axes.labelsize": 10,
        "legend.frameon": False, "legend.fontsize": 9, "font.size": 10,
        "axes.prop_cycle": plt.cycler(color=PALETTE),
    })


def save(fig: plt.Figure, name: str, outdir: str = "outputs/figures") -> str:
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"{name}.png")
    fig.savefig(path, bbox_inches="tight")
    return path


# --------------------------------------------------------------------------- #
def plot_growth(port: pd.Series, bench: pd.Series, bh: pd.Series, bench_name: str) -> plt.Figure:
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                                  gridspec_kw={"height_ratios": [3, 1.2]})
    for s, c, lw in [(port, PORT_COLOR, 2.0), (bench, BENCH_COLOR, 1.5), (bh, BH_COLOR, 1.2)]:
        ax.plot(s.index, (1 + s).cumprod(), lw=lw, color=c, label=s.name)
    ax.set_yscale("log"); ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v:,.1f}"))
    ax.set_title("Growth of $1"); ax.legend(loc="upper left")
    ax2.fill_between(port.index, met.drawdown_series(port), 0, color=PORT_COLOR, alpha=0.35, label="Portfolio")
    ax2.plot(bench.index, met.drawdown_series(bench), color=BENCH_COLOR, lw=1, label=bench_name)
    ax2.yaxis.set_major_formatter(_pct); ax2.set_title("Drawdown"); ax2.legend(loc="lower left")
    fig.tight_layout()
    return fig


def plot_rolling(rolling: pd.DataFrame, bench_name: str, window: int, long_window: int) -> plt.Figure:
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    axes[0].plot(rolling.index, rolling["vol"], color=PORT_COLOR, lw=1.5, label="Portfolio")
    axes[0].plot(rolling.index, rolling["bench_vol"], color=BENCH_COLOR, lw=1.2, label=bench_name)
    axes[0].yaxis.set_major_formatter(_pct); axes[0].set_title(f"Rolling {window}-day annualised volatility"); axes[0].legend()
    axes[1].plot(rolling.index, rolling["beta"], color=PORT_COLOR, lw=1.5, label="Beta")
    axes[1].plot(rolling.index, rolling["corr"], color=PALETTE[3], lw=1.2, label="Correlation")
    axes[1].axhline(1, color="grey", lw=0.8, ls=":"); axes[1].set_title(f"Rolling {window}-day beta and correlation to {bench_name}"); axes[1].legend()
    axes[2].plot(rolling.index, rolling["sharpe_3y"], color=PORT_COLOR, lw=1.5, label="Portfolio")
    axes[2].plot(rolling.index, rolling["bench_sharpe_3y"], color=BENCH_COLOR, lw=1.2, label=bench_name)
    axes[2].axhline(0, color="grey", lw=0.8); axes[2].set_title(f"Rolling {long_window}-day Sharpe ratio"); axes[2].legend()
    fig.tight_layout()
    return fig


def plot_risk_vs_weight(decomp: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(decomp)); w = 0.38
    ax.bar(x - w / 2, decomp["weight"], w, label="Capital weight", color=BENCH_COLOR)
    ax.bar(x + w / 2, decomp["pct_of_risk"], w, label="Share of portfolio risk", color=PORT_COLOR)
    for i, (wt, rk) in enumerate(zip(decomp["weight"], decomp["pct_of_risk"])):
        ax.text(i + w / 2, rk + 0.005, f"{rk / wt:.1f}x" if wt > 0 else "", ha="center", fontsize=8, color=PALETTE[1])
    ax.set_xticks(x); ax.set_xticklabels(decomp.index); ax.yaxis.set_major_formatter(_pct)
    ax.set_title(f"Weight vs risk contribution  (portfolio vol {decomp.attrs['portfolio_vol']:.1%}, "
                 f"diversification ratio {decomp.attrs['diversification_ratio']:.2f})")
    ax.legend()
    return fig


def plot_group_risk(grp: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, col, title in [(axes[0], "weight", "Capital allocation"), (axes[1], "pct_of_risk", "Risk allocation")]:
        vals = grp[col]
        ax.pie(vals, labels=vals.index, autopct="%1.0f%%", startangle=90, colors=PALETTE[:len(vals)],
               wedgeprops=dict(width=0.45, edgecolor="white"), textprops=dict(fontsize=8))
        ax.set_title(title); ax.grid(False)
    fig.tight_layout()
    return fig


def plot_rolling_risk(rr: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.stackplot(rr.index, [rr[c].values for c in rr.columns], labels=rr.columns,
                 colors=PALETTE[:len(rr.columns)], alpha=0.9)
    ax.set_ylim(0, 1); ax.yaxis.set_major_formatter(_pct); ax.margins(x=0)
    ax.set_title("Share of portfolio risk over time (drifted holdings, trailing 1-year covariance)")
    ax.legend(ncol=len(rr.columns), fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.08))
    return fig


def plot_monthly_heatmap(tbl: pd.DataFrame) -> plt.Figure:
    data = tbl.drop(columns=["Year"])
    fig, ax = plt.subplots(figsize=(11, 0.38 * len(data) + 1.5))
    vmax = np.nanmax(np.abs(data.values))
    im = ax.imshow(data.values, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(data.shape[1])); ax.set_xticklabels(data.columns)
    ax.set_yticks(range(len(data))); ax.set_yticklabels(data.index); ax.grid(False)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.1%}", ha="center", va="center", fontsize=7)
    for i, y in enumerate(tbl["Year"]):
        ax.text(data.shape[1] - 0.3, i, f"  {y:+.1%}", va="center", fontsize=8, fontweight="bold")
    ax.set_title("Monthly returns (year total at right)")
    return fig


def plot_annual(annual: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(11, 4))
    annual.plot.bar(ax=ax, width=0.8, color=[PORT_COLOR, BENCH_COLOR])
    ax.yaxis.set_major_formatter(_pct); ax.set_xlabel(""); ax.set_title("Calendar-year returns")
    plt.setp(ax.get_xticklabels(), rotation=0)
    return fig


def plot_correlation(corr: pd.DataFrame, title: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr))); ax.set_yticks(range(len(corr)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=8); ax.set_yticklabels(corr.index, fontsize=8); ax.grid(False)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(corr.values[i, j]) > 0.6 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046); ax.set_title(title)
    return fig


def plot_regime_correlation(to_bench: pd.DataFrame, bench_name: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(len(to_bench)); w = 0.38
    ax.bar(x - w / 2, to_bench["calm"], w, label="Calm days", color=BENCH_COLOR)
    ax.bar(x + w / 2, to_bench["stress"], w, label="Benchmark worst 10% days", color=PALETTE[1])
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(to_bench.index); ax.set_ylim(-1, 1)
    ax.set_title(f"Correlation to {bench_name}: calm vs stress days"); ax.legend()
    return fig


def plot_stress(stress: pd.DataFrame, bench_name: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(11, 4.5))
    x = np.arange(len(stress)); w = 0.38
    ax.bar(x - w / 2, stress["portfolio"], w, label="Portfolio (today's weights)", color=PORT_COLOR)
    ax.bar(x + w / 2, stress["benchmark"], w, label=bench_name, color=BENCH_COLOR)
    ax.set_xticks(x); ax.set_xticklabels(stress.index, rotation=25, ha="right", fontsize=8)
    ax.yaxis.set_major_formatter(_pct); ax.axhline(0, color="black", lw=0.8)
    ax.set_title("Historical stress scenarios"); ax.legend()
    fig.tight_layout()
    return fig


def plot_return_distribution(port: pd.Series, bench: pd.Series, var_level: float) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 4.5))
    bins = np.linspace(min(port.min(), bench.min()), max(port.max(), bench.max()), 80)
    ax.hist(bench, bins=bins, alpha=0.5, color=BENCH_COLOR, label=bench.name, density=True)
    ax.hist(port, bins=bins, alpha=0.6, color=PORT_COLOR, label=port.name, density=True)
    v, cv = met.var_cvar(port, var_level)
    ax.axvline(-v, color=PALETTE[1], ls="--", lw=1.2, label=f"VaR {var_level:.0%} = {v:.2%}")
    ax.axvline(-cv, color=PALETTE[1], ls=":", lw=1.2, label=f"CVaR {var_level:.0%} = {cv:.2%}")
    ax.xaxis.set_major_formatter(_pct1); ax.set_title("Distribution of daily returns"); ax.legend()
    return fig


def plot_what_if(tbl: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(len(tbl)); w = 0.2
    ax.bar(x - 1.5 * w, tbl["weight_before"], w, label="Weight before", color="#b0bec5")
    ax.bar(x - 0.5 * w, tbl["weight_after"], w, label="Weight after", color="#546e7a")
    ax.bar(x + 0.5 * w, tbl["risk_share_before"], w, label="Risk share before", color="#90caf9")
    ax.bar(x + 1.5 * w, tbl["risk_share_after"], w, label="Risk share after", color=PORT_COLOR)
    ax.set_xticks(x); ax.set_xticklabels(tbl.index); ax.yaxis.set_major_formatter(_pct)
    ax.set_title(f"What-if: portfolio vol {tbl.attrs['vol_before']:.2%} → {tbl.attrs['vol_after']:.2%}")
    ax.legend(ncol=4, fontsize=8)
    return fig
