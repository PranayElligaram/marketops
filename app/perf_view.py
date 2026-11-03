"""Presentation helpers for the Historical Performance tab.

Pure data shaping and chart construction only: nothing here recalculates a financial figure. Every
number comes from the API response; this module formats, labels, filters for display and plots it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import altair as alt
import pandas as pd

PERIODS = ("1M", "3M", "6M", "1Y", "ALL")
_OFFSETS = {
    "1M": pd.DateOffset(months=1),
    "3M": pd.DateOffset(months=3),
    "6M": pd.DateOffset(months=6),
    "1Y": pd.DateOffset(years=1),
}
_SYMBOLS = {"GBP": "£", "USD": "$", "EUR": "€"}

# API status -> precise reconciliation wording. The API values and rules are unchanged.
RECON_LABEL = {
    "verified": "Reconciled",
    "partial": "Partially reconciled",
    "unverified": "Not reconciled",
    "unavailable": "Unavailable",
}
MAPPING_LABEL = {
    "verified": "Symbol checked against fills",
    "unverified": "Symbol not confirmed",
    "unavailable": "No price data",
}

METRIC_LABELS = {
    "portfolio_nav": "Portfolio value",
    "net_invested_capital": "Net invested capital",
    "net_gain_on_invested_capital": "Net investment gain",
    "time_weighted_return": "Total time-weighted return",
    "annualised_twr": "Annualised time-weighted return",
    "money_weighted_return_xirr": "Money-weighted return (XIRR)",
    "annualised_volatility": "Annualised volatility",
    "max_drawdown": "Maximum drawdown",
    "twr_flow_timing_sensitivity": "Flow-timing sensitivity",
    "total_external_deposits": "External deposits",
    "total_external_withdrawals": "External withdrawals",
    "dividends_received": "Dividends received",
    "interest_received": "Interest received",
    "account_fees": "Account fees",
    "realised_pnl": "Realised P/L",
    "unrealised_pnl": "Unrealised P/L (live snapshot)",
    "portfolio_equity_curve": "Portfolio equity curve",
    "total_return_pct": "Total return %",
    "benchmark_comparison": "Benchmark comparison",
}

KPI_HELP = {
    "portfolio_nav": "Cash plus holdings valued at daily closing prices on the last valued day.",
    "net_invested_capital": "Deposits minus withdrawals. Deposits are never counted as profit.",
    "net_gain_on_invested_capital": "Portfolio value minus net invested capital.",
    "time_weighted_return": (
        "Compounded daily returns with deposits and withdrawals removed. Measures how the "
        "investments performed, independent of when money was added."
    ),
    "annualised_twr": "Time-weighted return scaled to a yearly rate; needs at least a year.",
    "money_weighted_return_xirr": (
        "Yearly rate that discounts your dated deposits and withdrawals to the live account "
        "value. Measures how your money performed, including the timing of your own deposits."
    ),
    "annualised_volatility": "Standard deviation of daily time-weighted returns, annualised.",
    "max_drawdown": "Largest peak-to-trough fall of the time-weighted index.",
}

KPI_ROWS: tuple[tuple[str, ...], ...] = (
    (
        "portfolio_nav",
        "net_invested_capital",
        "net_gain_on_invested_capital",
        "time_weighted_return",
    ),
    (
        "annualised_twr",
        "money_weighted_return_xirr",
        "annualised_volatility",
        "max_drawdown",
    ),
)

Tone = Literal["ok", "unreconciled", "unavailable", "pending"]


@dataclass(frozen=True)
class Kpi:
    key: str
    label: str
    value: str
    tone: Tone
    source: str
    note: str
    help: str


def label_for(name: str) -> str:
    return METRIC_LABELS.get(name, name.replace("_", " ").capitalize())


def symbol_for(ccy: str | None) -> str:
    return _SYMBOLS.get(ccy or "", f"{ccy} " if ccy else "")


def money(value: float, ccy: str | None) -> str:
    sign = "-" if value < 0 else ""
    return f"{sign}{symbol_for(ccy)}{abs(value):,.2f}"


def pct(value: float, signed: bool = False) -> str:
    return f"{value:+.2%}" if signed else f"{value:.2%}"


def _format(name: str, value: float, ccy: str | None) -> str:
    if name in ("portfolio_nav", "net_invested_capital"):
        return money(value, ccy)
    if name == "net_gain_on_invested_capital":
        return ("+" if value > 0 else "") + money(value, ccy)
    if name in ("time_weighted_return", "annualised_twr", "max_drawdown"):
        return pct(value, signed=name != "max_drawdown")
    return pct(value)


def build_kpis(val: dict[str, Any] | None, perf: dict[str, Any] | None) -> list[list[Kpi]]:
    """Two rows of four KPIs. Values are taken as given; none is computed here."""
    ccy = (val or {}).get("account_currency") or (perf or {}).get("account_currency")
    by_name = {m["name"]: m for m in (val or {}).get("metrics", [])}
    rows: list[list[Kpi]] = []
    for names in KPI_ROWS:
        row: list[Kpi] = []
        for name in names:
            label, help_ = label_for(name), KPI_HELP.get(name, "")
            if name == "money_weighted_return_xirr":
                row.append(_xirr_kpi(perf, label, help_))
                continue
            if val is None:
                row.append(Kpi(name, label, "—", "pending", "Historical closes", "", help_))
                continue
            m = by_name.get(name)
            if m is None or m["value"] is None:
                blockers = (m or {}).get("blockers") or ["Not available."]
                row.append(
                    Kpi(name, label, "—", "unavailable", "Historical closes", blockers[0], help_)
                )
                continue
            ok = m["status"] == "available"
            row.append(
                Kpi(
                    name,
                    label,
                    _format(name, m["value"], ccy),
                    "ok" if ok else "unreconciled",
                    "Historical closes",
                    "" if ok else "Not reconciled to the live account.",
                    help_,
                )
            )
        rows.append(row)
    return rows


def _xirr_kpi(perf: dict[str, Any] | None, label: str, help_: str) -> Kpi:
    source = "Live snapshot"
    if perf is None:
        return Kpi(
            "money_weighted_return_xirr",
            label,
            "—",
            "pending",
            source,
            "Press “Calculate historical performance” (reads the live account).",
            help_,
        )
    m = next((x for x in perf["metrics"] if x["name"] == "money_weighted_return_xirr"), None)
    if m is None or m["status"] != "available" or not m.get("value"):
        blockers = (m or {}).get("blockers") or ["Not available."]
        return Kpi(
            "money_weighted_return_xirr", label, "—", "unavailable", source, blockers[0], help_
        )
    return Kpi(
        "money_weighted_return_xirr",
        label,
        f"{m['value']['annualised_rate']:+.2%} / yr",
        "ok",
        source,
        "",
        help_,
    )


# ------------------------------------------------------------------ series and tables
def series_frame(val: dict[str, Any]) -> pd.DataFrame:
    """Full-history chart frame. Cumulative TWR and drawdown are derived from the API's own
    time-weighted index over the whole history so a display window never changes them."""
    s = pd.DataFrame(val["series"])
    if s.empty:
        return pd.DataFrame(columns=["date", "nav", "net_invested", "twr", "drawdown"])
    s["date"] = pd.to_datetime(s["date"])
    idx = s["twr_index"].astype(float)
    s["twr"] = idx - 1.0
    s["drawdown"] = idx / idx.cummax() - 1.0
    # Display only: drop the leading days before the first valuation so the axis starts at the data.
    first = s["nav"].first_valid_index()
    s = s if first is None else s[s.index >= first]
    return s[["date", "nav", "net_invested", "twr", "drawdown"]].reset_index(drop=True)


def filter_period(df: pd.DataFrame, period: str) -> pd.DataFrame:
    """Display window ending at the last date in the data; ALL returns everything."""
    if df.empty or period == "ALL" or period not in _OFFSETS:
        return df
    cutoff = df["date"].max() - _OFFSETS[period]
    return pd.DataFrame(df[df["date"] >= cutoff].reset_index(drop=True))


def period_table(rows: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=["period", "twr", "days", "complete"])
    df = df.sort_values("period", ascending=False)
    df["complete"] = df["complete"].map({True: "Complete", False: "Partial period"})
    return df.rename(
        columns={"period": "Period", "twr": "Return", "days": "Days", "complete": "Coverage"}
    ).reset_index(drop=True)


def style_period_table(df: pd.DataFrame) -> Any:
    def colour(v: Any) -> str:
        return "color: #1b7f4c" if v > 0 else "color: #c0392b" if v < 0 else ""

    return df.style.format({"Return": "{:+.2%}"}).map(colour, subset=["Return"])


# ------------------------------------------------------------------ charts
def _axis_labels(ccy: str | None) -> str:
    sym = symbol_for(ccy).strip().replace("'", "")
    return f"'{sym}' + format(datum.value, ',.0f')" if sym else "format(datum.value, ',.0f')"


def _hover(data: pd.DataFrame, y: str, tips: list[alt.Tooltip]) -> list[alt.Chart]:
    """Invisible points plus a nearest-date rule so the tooltip follows the pointer."""
    sel = alt.selection_point(
        fields=["date"], nearest=True, on="pointerover", empty=False, clear="pointerout"
    )
    base = alt.Chart(data)
    pick = base.mark_point(opacity=0).encode(x="date:T", y=y).add_params(sel)
    rule = (
        base.mark_rule(color="#888", strokeDash=[3, 3])
        .encode(x="date:T", tooltip=tips)
        .transform_filter(sel)
    )
    return [pick, rule]


def value_chart(df: pd.DataFrame, ccy: str | None, height: int = 340) -> alt.TopLevelMixin:
    long = df.melt(
        id_vars="date",
        value_vars=["nav", "net_invested"],
        var_name="series",
        value_name="value",
    )
    long["series"] = long["series"].map(
        {"nav": "Portfolio value", "net_invested": "Net invested capital"}
    )
    colours = alt.Scale(
        domain=["Portfolio value", "Net invested capital"], range=["#2a6df4", "#9aa3b2"]
    )
    base = alt.Chart(long)
    lines = base.mark_line(strokeWidth=2).encode(
        x=alt.X("date:T", title=None, axis=alt.Axis(format="%b %Y", labelOverlap=True)),
        y=alt.Y(
            "value:Q",
            title=f"Value ({ccy})" if ccy else "Value",
            scale=alt.Scale(zero=False),
            axis=alt.Axis(labelExpr=_axis_labels(ccy)),
        ),
        color=alt.Color("series:N", scale=colours, legend=alt.Legend(title=None, orient="top")),
        strokeDash=alt.StrokeDash(
            "series:N",
            scale=alt.Scale(
                domain=["Portfolio value", "Net invested capital"], range=[[1, 0], [5, 3]]
            ),
            legend=None,
        ),
    )
    wide = df.rename(columns={"nav": "Portfolio value", "net_invested": "Net invested capital"})
    sym = symbol_for(ccy)
    tips = [
        alt.Tooltip("date:T", title="Date", format="%d %b %Y"),
        alt.Tooltip("Portfolio value:Q", format=f"{sym},.2f"),
        alt.Tooltip("Net invested capital:Q", format=f"{sym},.2f"),
    ]
    wide = df.rename(columns={"nav": "Portfolio value", "net_invested": "Net invested capital"})
    return alt.layer(lines, *_hover(wide, "Portfolio value:Q", tips)).properties(height=height)


def _pct_chart(
    df: pd.DataFrame, col: str, title: str, colour: str, area: bool, height: int
) -> alt.TopLevelMixin:
    d = df[["date", col]].rename(columns={col: title})
    enc_y = alt.Y(
        f"{title}:Q",
        title=title,
        axis=alt.Axis(format="%"),
        scale=alt.Scale(zero=area),
    )
    x = alt.X("date:T", title=None, axis=alt.Axis(format="%b %Y", labelOverlap=True))
    base = alt.Chart(d)
    mark = (
        base.mark_area(color=colour, opacity=0.35, line={"color": colour})
        if area
        else base.mark_line(color=colour, strokeWidth=2)
    )
    tips = [
        alt.Tooltip("date:T", title="Date", format="%d %b %Y"),
        alt.Tooltip(f"{title}:Q", format="+.2%"),
    ]
    return alt.layer(mark.encode(x=x, y=enc_y), *_hover(d, f"{title}:Q", tips)).properties(
        height=height
    )


def twr_chart(df: pd.DataFrame, height: int = 260) -> alt.TopLevelMixin:
    return _pct_chart(df, "twr", "Cumulative time-weighted return", "#1b7f4c", False, height)


def drawdown_chart(df: pd.DataFrame, height: int = 200) -> alt.TopLevelMixin:
    return _pct_chart(df, "drawdown", "Drawdown", "#c0392b", True, height)


# ------------------------------------------------------------------ benchmark comparison
BENCH = "S&P 500 benchmark"
PORT = "Portfolio"
NETINV = "Net invested capital"
CMP_COLOURS = {PORT: "#2a6df4", BENCH: "#e0892b", NETINV: "#9aa3b2"}


def pp(value: float) -> str:
    """Percentage-point difference, e.g. +7.74 pp."""
    return f"{value * 100:+.2f} pp"


def _cmp_format(name: str, value: float, ccy: str | None) -> str:
    if name == "value_difference":
        return ("+" if value > 0 else "") + money(value, ccy)
    if name.endswith("_value") or name == "net_invested_capital":
        return money(value, ccy)
    if name.endswith("_difference"):
        return pp(value) + (" / yr" if name in ("annualised_difference", "xirr_difference") else "")
    if "xirr" in name or "annualised" in name:
        return f"{value:+.2%} / yr"
    if name.endswith("drawdown"):
        return pct(value)
    return pct(value, signed=True)


CMP_LABELS = {
    "portfolio_twr": "Portfolio return (TWR)",
    "benchmark_twr": "S&P 500 return (TWR)",
    "twr_difference": "Portfolio vs S&P 500",
    "value_difference": "Value vs S&P 500",
    "portfolio_annualised_twr": "Portfolio annualised",
    "benchmark_annualised_twr": "S&P 500 annualised",
    "annualised_difference": "Annualised difference",
    "portfolio_max_drawdown": "Portfolio max drawdown",
    "benchmark_max_drawdown": "S&P 500 max drawdown",
    "drawdown_difference": "Drawdown difference",
    "portfolio_xirr_closing": "Portfolio XIRR (to last close)",
    "benchmark_xirr_closing": "S&P 500 XIRR (same flows)",
    "xirr_difference": "XIRR difference",
    "benchmark_value": "Same money in the S&P 500",
    "portfolio_value": "Portfolio value",
    "net_invested_capital": NETINV,
}
CMP_HELP = {
    "twr_difference": "Portfolio time-weighted return minus the benchmark, in percentage points.",
    "value_difference": "Portfolio value minus what the same deposits and withdrawals would be "
    "worth in the benchmark today. Positive = ahead of the benchmark.",
    "benchmark_value": "Every deposit bought, and every withdrawal sold, in the benchmark at the "
    "close of the same day.",
    "portfolio_xirr_closing": "XIRR on your dated flows ending at the last common closing value. "
    "It is not the live-snapshot XIRR above, which ends at a live balance.",
    "drawdown_difference": "Portfolio minus benchmark maximum drawdown; negative means the "
    "portfolio fell further.",
}
CMP_ROWS: tuple[tuple[str, ...], ...] = (
    ("portfolio_twr", "benchmark_twr", "twr_difference", "value_difference"),
    (
        "portfolio_annualised_twr",
        "benchmark_annualised_twr",
        "portfolio_max_drawdown",
        "benchmark_max_drawdown",
    ),
    ("portfolio_xirr_closing", "benchmark_xirr_closing", "xirr_difference", "benchmark_value"),
)
CMP_COLUMNS = [
    "date",
    "portfolio_value",
    "benchmark_value",
    "net_invested",
    "portfolio_twr",
    "benchmark_twr",
    "portfolio_drawdown",
    "benchmark_drawdown",
]


def build_comparison_kpis(bench: dict[str, Any]) -> list[list[Kpi]]:
    """Comparative KPI rows from the API's benchmark block. Nothing is computed here."""
    ccy = bench.get("currency")
    by_name = {m["name"]: m for m in bench.get("metrics", [])}
    rows: list[list[Kpi]] = []
    for names in CMP_ROWS:
        row: list[Kpi] = []
        for name in names:
            label, help_ = CMP_LABELS[name], CMP_HELP.get(name, "")
            m = by_name.get(name)
            if m is None or m["value"] is None:
                why = ((m or {}).get("blockers") or ["Not available."])[0]
                row.append(Kpi(name, label, "—", "unavailable", "Benchmark comparison", why, help_))
                continue
            ok = m["status"] == "available"
            row.append(
                Kpi(
                    name,
                    label,
                    _cmp_format(name, m["value"], ccy),
                    "ok" if ok else "unreconciled",
                    "Benchmark comparison",
                    "" if ok else "Portfolio not reconciled.",
                    help_,
                )
            )
        rows.append(row)
    return rows


def comparison_frame(bench: dict[str, Any]) -> pd.DataFrame:
    df = pd.DataFrame(bench.get("series", []), columns=CMP_COLUMNS)
    df["date"] = pd.to_datetime(df["date"])
    return df


def window_comparison(df: pd.DataFrame, period: str) -> pd.DataFrame:
    """Display window with cumulative returns rebased to 0 at the window start.

    Rebasing (1+r_t)/(1+r_0)-1 is exact for chain-linked returns, so a window figure equals the
    return of that window. Drawdowns stay as measured from the full-history peak.
    """
    shown = filter_period(df, period)
    if shown.empty:
        return shown
    shown = shown.copy()
    for col in ("portfolio_twr", "benchmark_twr"):
        shown[col] = (1.0 + shown[col]) / (1.0 + shown[col].iloc[0]) - 1.0
    return shown


def _line_chart(
    wide: pd.DataFrame,
    domain: list[str],
    y: alt.Y,
    tip_fmt: str,
    dashes: list[list[int]],
    height: int,
) -> alt.TopLevelMixin:
    long = wide.melt(id_vars="date", var_name="series", value_name="value")
    colour = alt.Color(
        "series:N",
        scale=alt.Scale(domain=domain, range=[CMP_COLOURS[d] for d in domain]),
        legend=alt.Legend(title=None, orient="top"),
    )
    lines = (
        alt.Chart(long)
        .mark_line(strokeWidth=2)
        .encode(
            x=alt.X("date:T", title=None, axis=alt.Axis(format="%b %Y", labelOverlap=True)),
            y=y,
            color=colour,
            strokeDash=alt.StrokeDash(
                "series:N", scale=alt.Scale(domain=domain, range=dashes), legend=None
            ),
        )
    )
    tips = [alt.Tooltip("date:T", title="Date", format="%d %b %Y")] + [
        alt.Tooltip(f"{n}:Q", format=tip_fmt) for n in domain
    ]
    return alt.layer(lines, *_hover(wide, f"{domain[0]}:Q", tips)).properties(height=height)


def comparison_value_chart(
    df: pd.DataFrame, ccy: str | None, height: int = 340
) -> alt.TopLevelMixin:
    names = {"portfolio_value": PORT, "benchmark_value": BENCH, "net_invested": NETINV}
    y = alt.Y(
        "value:Q",
        title=f"Value ({ccy})" if ccy else "Value",
        scale=alt.Scale(zero=False),
        axis=alt.Axis(labelExpr=_axis_labels(ccy)),
    )
    return _line_chart(
        df[["date", *names]].rename(columns=names),
        [PORT, BENCH, NETINV],
        y,
        f"{symbol_for(ccy)},.2f",
        [[1, 0], [1, 0], [5, 3]],
        height,
    )


def _pct_two(
    df: pd.DataFrame, cols: tuple[str, str], title: str, fmt: str, height: int
) -> alt.TopLevelMixin:
    wide = df[["date", *cols]].rename(columns={cols[0]: PORT, cols[1]: BENCH})
    y = alt.Y("value:Q", title=title, axis=alt.Axis(format="%"), scale=alt.Scale(zero=False))
    return _line_chart(wide, [PORT, BENCH], y, fmt, [[1, 0], [1, 0]], height)


def comparison_twr_chart(df: pd.DataFrame, height: int = 260) -> alt.TopLevelMixin:
    return _pct_two(
        df, ("portfolio_twr", "benchmark_twr"), "Return since window start", "+.2%", height
    )


def comparison_drawdown_chart(df: pd.DataFrame, height: int = 220) -> alt.TopLevelMixin:
    return _pct_two(
        df,
        ("portfolio_drawdown", "benchmark_drawdown"),
        "Drawdown from previous peak",
        ".2%",
        height,
    )
