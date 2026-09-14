"""
Single-file HTML dashboard.

Figures are embedded as base64 PNGs so the report has no external
dependencies and can be e-mailed or dropped into a shared drive as-is.
"""
from __future__ import annotations

import base64
import io
from datetime import date
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd

from . import metrics as met

_CSS = """
body{font-family:-apple-system,Segoe UI,Helvetica,Arial,sans-serif;margin:0;background:#f5f6f8;color:#222}
.wrap{max-width:1180px;margin:0 auto;padding:28px}
h1{margin:0 0 4px;font-size:26px;color:#1f4e79} h2{font-size:18px;margin:34px 0 10px;color:#1f4e79;border-bottom:2px solid #dfe3e8;padding-bottom:4px}
.sub{color:#666;margin-bottom:18px}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px}
.kpi{background:#fff;border-radius:8px;padding:12px 14px;box-shadow:0 1px 3px rgba(0,0,0,.08)}
.kpi .l{font-size:11px;color:#777;text-transform:uppercase;letter-spacing:.04em}
.kpi .v{font-size:22px;font-weight:600;margin-top:2px} .kpi .b{font-size:11px;color:#999}
.card{background:#fff;border-radius:8px;padding:14px;box-shadow:0 1px 3px rgba(0,0,0,.08);margin-top:12px}
img{max-width:100%;height:auto;display:block;margin:0 auto}
table{border-collapse:collapse;width:100%;font-size:12.5px} th,td{padding:6px 8px;border-bottom:1px solid #eee;text-align:right}
th{background:#f0f3f7;color:#333} td:first-child,th:first-child{text-align:left}
.note{font-size:12px;color:#777;margin-top:8px}
"""


def _fig_b64(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _kpi(label: str, value: str, sub: str = "") -> str:
    return f'<div class="kpi"><div class="l">{label}</div><div class="v">{value}</div><div class="b">{sub}</div></div>'


def _table(df: pd.DataFrame) -> str:
    return df.to_html(border=0, classes="tbl", escape=False)


def build_report(R: Dict[str, object], figures: List[Tuple[str, plt.Figure]],
                 title: str, out_path: str, rebalance_label: str) -> str:
    s = R["summary"]
    port, bench = s["Portfolio"], s[R["benchmark_name"]]
    dec = R["decomposition"]
    top_risk = dec["pct_of_risk"].idxmax()

    kpis = "".join([
        _kpi("CAGR", f"{port['CAGR']:.2%}", f"benchmark {bench['CAGR']:.2%}"),
        _kpi("Volatility", f"{port['Volatility']:.2%}", f"benchmark {bench['Volatility']:.2%}"),
        _kpi("Sharpe", f"{port['Sharpe']:.2f}", f"benchmark {bench['Sharpe']:.2f}"),
        _kpi("Max drawdown", f"{port['Max drawdown']:.1%}", f"{port['Max DD length (days)']:.0f} days peak to recovery"),
        _kpi("Beta", f"{port['Beta']:.2f}", f"tracking error {port['Tracking error']:.2%}"),
        _kpi("Info ratio", f"{port['Information ratio']:.2f}", f"alpha {port['Alpha (ann)']:+.2%} (t={port['Alpha t-stat']:.1f})"),
        _kpi("Daily CVaR 95%", f"{port['Daily CVaR 95%']:.2%}", f"VaR {port['Daily VaR 95%']:.2%}"),
        _kpi("Largest risk", top_risk, f"{dec.loc[top_risk, 'pct_of_risk']:.0%} of risk from {dec.loc[top_risk, 'weight']:.0%} weight"),
    ])

    parts = [f"<html><head><meta charset='utf-8'><title>{title}</title><style>{_CSS}</style></head><body><div class='wrap'>",
             f"<h1>{title}</h1><div class='sub'>{R['portfolio'].index[0].date()} → {R['portfolio'].index[-1].date()} · "
             f"{rebalance_label} · benchmark {R['benchmark_name']} · generated {date.today()}</div>",
             f"<div class='kpis'>{kpis}</div>"]

    for heading, fig in figures:
        parts.append(f"<h2>{heading}</h2><div class='card'><img src='data:image/png;base64,{_fig_b64(fig)}'/></div>")

    parts.append("<h2>Performance summary</h2><div class='card'>" + _table(met.format_summary(s)) + "</div>")

    d = dec.copy()
    for c in ["weight", "stand_alone_vol", "marginal_ctr", "component_ctr", "pct_of_risk"]:
        d[c] = d[c].map(lambda x: f"{x:.2%}")
    d["beta_to_portfolio"] = d["beta_to_portfolio"].map(lambda x: f"{x:.2f}")
    d["risk_to_weight"] = d["risk_to_weight"].map(lambda x: f"{x:.2f}x")
    parts.append("<h2>Risk decomposition (Euler)</h2><div class='card'>" + _table(d) +
                 f"<div class='note'>Portfolio vol {dec.attrs['portfolio_vol']:.2%} · diversification ratio "
                 f"{dec.attrs['diversification_ratio']:.2f} · effective number of risk sources {dec.attrs['effective_n_risk']:.1f}</div></div>")

    dd = R["drawdowns"].copy()
    dd["depth"] = dd["depth"].map(lambda x: f"{x:.1%}")
    parts.append("<h2>Largest drawdowns</h2><div class='card'>" + _table(dd) + "</div>")

    st = R["stress"].copy()
    for c in ["portfolio", "benchmark", "worst_day", "worst_asset_return"]:
        st[c] = st[c].map(lambda x: f"{x:.1%}")
    parts.append("<h2>Historical stress scenarios</h2><div class='card'>" + _table(st) +
                 "<div class='note'>Today's weights held buy-and-hold through each window.</div></div>")

    parts.append("</div></body></html>")
    html = "\n".join(parts)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    return out_path
