"""MarketOps dashboard: read-only portfolio view plus ML research results from the API."""

import os
from typing import Any

import httpx
import pandas as pd
import perf_view as pv
import streamlit as st

st.set_page_config(page_title="MarketOps", page_icon="📈", layout="wide")
API = os.getenv("MARKETOPS_API_BASE_URL", "http://localhost:8001").rstrip("/")


def _private_headers(key: str | None) -> dict[str, str] | None:
    """Token typed by the user this session; never read from the environment, so the
    dashboard container holds no secret and cannot act as an unauthenticated proxy."""
    token = st.session_state.get(key, "") if key else ""
    return {"X-MarketOps-Token": token} if token else None


def post(
    path: str, body: dict[str, Any], timeout: float = 330.0, token_key: str | None = None
) -> Any:
    """POST JSON to the API; returns (data, error_message)."""
    try:
        headers = _private_headers(token_key)
        r = httpx.post(f"{API}{path}", json=body, headers=headers, timeout=timeout)
        if r.status_code not in (200, 202):
            try:
                return None, r.json().get("detail", f"HTTP {r.status_code}")
            except ValueError:
                return None, f"HTTP {r.status_code}"
        return r.json(), None
    except httpx.TimeoutException:
        return None, "The research request timed out."
    except (httpx.RequestError, ValueError):
        return None, "Cannot reach the MarketOps API."


def get(path: str, token_key: str | None = None, **params: Any) -> Any:
    """GET from the API; returns (data, error_message)."""
    try:
        headers = _private_headers(token_key)
        r = httpx.get(f"{API}{path}", params=params or None, headers=headers, timeout=30.0)
        if r.status_code not in (200, 202):
            try:
                return None, r.json().get("detail", f"HTTP {r.status_code}")
            except ValueError:
                return None, f"HTTP {r.status_code}"
        return r.json(), None
    except (httpx.RequestError, ValueError):
        return None, "Cannot reach the MarketOps API."


def _metric_text(m: dict[str, Any]) -> str:
    v = m.get("value") or {}
    ccy = m.get("unit") or ""
    if m["name"] == "money_weighted_return_xirr":
        return f"{v['annualised_rate'] * 100:.2f}% per year"
    parts = [f"{k.replace('_', ' ')} {x:,.2f} {ccy}".strip() for k, x in v.items() if k != "count"]
    return "; ".join(parts)


def _kpi_grid(val: dict[str, Any], perf: dict[str, Any] | None) -> None:
    for row in pv.build_kpis(val, perf):
        for col, k in zip(st.columns(len(row)), row, strict=True):
            with col.container(border=True):
                st.metric(k.label, k.value, help=k.help or None)
                st.caption(k.source + (f" · {k.note}" if k.note else ""))
    st.caption(
        "**Time-weighted return (TWR)** removes the effect of when you added or withdrew money, "
        "so it shows how the investments performed. **XIRR** is the yearly rate on your actual "
        "dated cash flows, so it also reflects the timing of your own deposits. They answer "
        "different questions and are not expected to match."
    )


def _period_tables(val: dict[str, Any]) -> None:
    st.markdown("**Monthly and annual time-weighted return**")
    left, right = st.columns([3, 2])
    for col, name, key in ((left, "Monthly", "monthly"), (right, "Annual", "annual")):
        with col:
            if not val[key]:
                st.caption(f"{name}: no valuation periods yet.")
                continue
            table = pv.period_table(val[key])
            st.dataframe(
                pv.style_period_table(table),
                hide_index=True,
                width="stretch",
                height=min(38 + 35 * len(table), 300),
            )
    st.caption("Partial periods contain fewer valued days than the full period.")


def _charts(val: dict[str, Any]) -> None:
    frame = pv.series_frame(val)
    if frame.empty or frame["nav"].notna().sum() == 0:
        st.info("No valued days yet, so there is nothing to chart. See data coverage below.")
        return
    period = st.segmented_control("Period", pv.PERIODS, default="ALL", key="perf_period") or "ALL"
    shown = pv.filter_period(frame, period)
    st.markdown("**Portfolio value vs net invested capital**")
    st.altair_chart(pv.value_chart(shown, val["account_currency"]), width="stretch")
    st.caption(
        f"Showing {shown['date'].min():%d %b %Y} to {shown['date'].max():%d %b %Y}. Return and "
        "drawdown are measured from the start of the full history, so the period selector only "
        "changes what is displayed, never a calculated figure. Gaps mean no reconciled valuation."
    )
    if shown["twr"].notna().any():
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Cumulative time-weighted return**")
            st.altair_chart(pv.twr_chart(shown), width="stretch")
        with c2:
            st.markdown("**Drawdown from previous peak**")
            st.altair_chart(pv.drawdown_chart(shown), width="stretch")


def _valuation_details(val: dict[str, Any]) -> None:
    cov, rec = val["coverage"], val["reconciliation"]
    with st.expander("Reconciliation with the live account"):
        diff = (
            f" ({rec['difference_pct']:+.2%} on {rec['valuation_date']})"
            if rec["difference_pct"] is not None
            else ""
        )
        st.markdown(
            f"- Closing-price value vs live Trading 212 total: "
            f"**{rec['status'].replace('_', ' ')}**{diff}\n"
            f"- Share counts: **{rec['quantity_status'].replace('_', ' ')}**"
        )
        if rec["holdings"]:
            st.dataframe(pd.DataFrame(rec["holdings"]), hide_index=True, width="stretch")
    with st.expander("Data coverage"):
        fx = ", ".join(f"{c} {s.replace('_', ' ')}" for c, s in cov["fx_currencies"].items())
        st.markdown(
            f"- Range: {cov['first_day'] or '-'} to {cov['last_day'] or '-'}\n"
            f"- Weekdays valued: **{cov['valued_days']} of {cov['weekdays']}** "
            f"({cov['stale_price_days']} using a price older than the day)\n"
            f"- Exchange rates: {fx or '-'}"
        )
        if cov["unvalued_reasons"]:
            reasons = [
                {"Reason": k.replace("_", " "), "Weekdays": n}
                for k, n in cov["unvalued_reasons"].items()
            ]
            st.dataframe(pd.DataFrame(reasons), hide_index=True)
    with st.expander("Price-provider and instrument-mapping diagnostics"):
        if val["mapping"]:
            mapping = pd.DataFrame(val["mapping"])
            mapping["status"] = mapping["status"].map(lambda s: pv.MAPPING_LABEL.get(s, s))
            mapping["reasons"] = mapping["reasons"].map(" ".join)
            st.dataframe(mapping, hide_index=True, width="stretch")
        else:
            st.caption("No instruments.")
    with st.expander("Assumptions and conventions"):
        for k, v in val["conventions"].items():
            st.markdown(f"- **{k.replace('_', ' ')}**: {v}")


def _benchmark_explainer(bench: dict[str, Any]) -> None:
    with st.expander("How to read this comparison (XIRR vs time-weighted return)"):
        st.markdown(
            "- **The question answered:** if every pound you deposited had instead bought the "
            "S&P 500 (via the SPY fund, dividends reinvested, priced in £) at the close of the "
            "same day, and every withdrawal had sold it, what would that be worth today?\n"
            "- **Time-weighted return (TWR)** removes the effect of when money went in or out. "
            "It compares how the *investments* performed and is the fair test against an index.\n"
            "- **XIRR** is the yearly rate on your actual dated deposits and withdrawals, so it "
            "also reflects *when you chose to invest*. It can be far from the TWR when large "
            "deposits arrive late.\n"
            "- **Why the XIRR here differs from the live XIRR above:** the live figure ends at a "
            "live broker balance. The benchmark can only be valued at a closing price, so both "
            "XIRRs here use the same flows and end on the same last common close.\n"
            "- **Drawdown** is the fall from the previous high; the one with the smaller "
            "number was the smoother ride.\n"
            "- A positive gap does not prove skill, and the portfolio may simply hold different "
            "risks than the index."
        )


def _benchmark_details(bench: dict[str, Any]) -> None:
    src = bench.get("source") or {}
    with st.expander("Data quality and methodology"):
        if src:
            age = f"{src['age_days']} day(s) ago" if src.get("age_days") is not None else "unknown"
            st.markdown(
                f"- **Source:** {src['symbol']} via {src.get('provider') or 'unknown provider'}; "
                f"{src['basis']}\n"
                f"- **Benchmark days usable:** {src['rows_usable']} "
                f"({src['first_day']} to {src['last_day']}); dropped for invalid price: "
                f"{src['rows_dropped_invalid_price']}, for missing exchange rate: "
                f"{src['rows_dropped_missing_fx']}\n"
                f"- **Downloaded:** {src.get('retrieved_utc') or 'unknown'} ({age}); file "
                f"fingerprint `{(src.get('sha256') or 'none (synthetic)')[:16]}`\n"
                f"- **Comparison window:** {bench['window_start']} to {bench['window_end']}, "
                f"{bench['common_days']} days with both a portfolio value and a benchmark price"
            )
        for k, v in bench["conventions"].items():
            st.markdown(f"- **{k.replace('_', ' ')}**: {v}")
        st.markdown("**Limitations**")
        for lim in bench["limitations"]:
            st.markdown(f"- {lim}")


def _benchmark_section(val: dict[str, Any]) -> None:
    st.divider()
    st.subheader("Portfolio vs S&P 500 benchmark")
    bench = val.get("benchmark")
    if bench is None:
        st.info("The benchmark comparison is not part of this response.")
        return
    if bench["status"] == "unavailable":
        st.warning("**Comparison unavailable.** " + " ".join(bench["reasons"]))
        st.caption(
            "Use “Download prices, FX and S&P 500 data” after syncing history. Nothing is "
            "estimated or substituted when data is missing."
        )
        with st.expander("Methodology and limitations"):
            for lim in bench["limitations"]:
                st.markdown(f"- {lim}")
        return
    if bench["status"] == "partial":
        for r in bench["reasons"]:
            st.warning(r)
    st.caption(
        "What the same deposits and withdrawals would be worth if every pound had gone into the "
        "S&P 500 (SPY, dividends reinvested, in £) on the same day. "
        f"Compared {bench['window_start']} to {bench['window_end']}, closing prices only. The "
        "window starts on the first day both the portfolio and the benchmark have a price, so "
        "portfolio figures here can differ slightly from the headline cards above."
    )
    for row in pv.build_comparison_kpis(bench):
        for col, k in zip(st.columns(len(row)), row, strict=True):
            with col.container(border=True):
                st.metric(k.label, k.value, help=k.help or None)
                st.caption(k.source + (f" · {k.note}" if k.note else ""))
    frame = pv.comparison_frame(bench)
    period = st.segmented_control("Period", pv.PERIODS, default="ALL", key="bench_period") or "ALL"
    shown = pv.window_comparison(frame, period)
    st.markdown("**Portfolio value, S&P 500 benchmark value and net invested capital**")
    st.altair_chart(pv.comparison_value_chart(shown, bench["currency"]), width="stretch")
    st.caption(
        "The benchmark line is the same money invested in the S&P 500. Net invested capital is "
        "identical for both. The period selector changes the display only; headline figures "
        "above always cover the full comparison window."
    )
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Cumulative return since the start of the displayed period**")
        st.altair_chart(pv.comparison_twr_chart(shown), width="stretch")
    with c2:
        st.markdown("**Drawdown from previous peak**")
        st.altair_chart(pv.comparison_drawdown_chart(shown), width="stretch")
    _benchmark_explainer(bench)
    _benchmark_details(bench)


def _render_valuation(val: dict[str, Any], perf: dict[str, Any] | None = None) -> None:
    status, cov = val["overall_status"], val["coverage"]
    banner = {"verified": st.success, "partial": st.warning, "unavailable": st.info}[status]
    banner(
        f"**Historical closing-price valuation** ({val['account_currency'] or '?'}): "
        f"reconciliation status **{pv.RECON_LABEL[status]}**. Valued "
        f"{cov['first_day'] or '-'} to {cov['last_day'] or '-'}; calculated "
        f"{val['as_of_utc'][:19]} UTC."
    )
    st.caption(
        "Values use end-of-day closing prices and ECB reference exchange rates, so they can differ "
        "from the live Trading 212 snapshot, which uses current broker prices."
    )
    if any(n.startswith("SYNTHETIC") for n in val["notes"]):
        st.warning(
            "**Synthetic demonstration.** Every figure below is invented; no real account is read."
        )
    for code, n in cov["unvalued_reasons"].items():
        st.warning(f"{n} weekday(s) not valued: {code.replace('_', ' ')}.")
    for m in val["mapping"]:
        if m["status"] != "verified":
            st.warning(
                f"Instrument {m['ticker']}: {pv.MAPPING_LABEL.get(m['status'], m['status'])} "
                f"(symbol {m['symbol'] or 'none'}). " + " ".join(m["reasons"])
            )
    for r in val["reconciliation"]["reasons"]:
        st.warning(r)
    for m in val["metrics"]:
        if m["value"] is None and m["name"] not in pv.KPI_HELP:
            st.info(f"{pv.label_for(m['name'])}: unavailable. " + " ".join(m["blockers"][:2]))
    _kpi_grid(val, perf)
    for note in val["notes"]:
        st.caption(note)
    _charts(val)
    _benchmark_section(val)
    _period_tables(val)
    _valuation_details(val)


def _render_snapshot(perf: dict[str, Any], show_xirr: bool) -> None:
    status = perf["overall_status"]
    banner = {"verified": st.success, "partial": st.warning, "unverified": st.warning}[status]
    banner(
        f"**Live Trading 212 snapshot** ({perf['environment'].upper()} account, "
        f"{perf['account_currency'] or '?'}): reconciliation status **{pv.RECON_LABEL[status]}** "
        f"as of {perf['as_of_utc'][:19]} UTC."
    )
    st.caption(
        "Current positions and cash read from Trading 212, compared with the history rebuilt from "
        "stored orders and cash movements. XIRR uses this snapshot as its end value."
    )
    for note in perf["notes"]:
        (st.caption if status == "verified" else st.warning)(note)
    audit, hold, led = perf["data_audit"], perf["holdings"], perf["cash_ledger"]
    for k in audit["kinds"]:
        for c in k["checks"]:
            if c["ok"] is not True:
                st.warning(f"{k['kind']} - {c['name'].replace('_', ' ')}: {c['detail']}")
    for issue in audit["issues"]:
        st.warning(issue)
    for reason in led["reasons"]:
        st.warning(reason)
    r1, r2 = st.columns(2)
    r1.metric("Positions", hold["status"], f"{hold['matched']} of {hold['checked']} match")
    r2.metric("Cash ledger", led["status"])
    if led["difference"] is not None:
        st.caption(
            f"Cash implied by history minus broker cash: {led['difference']:+,.2f} "
            f"(tolerance {led['tolerance']:,.2f})."
        )
    with st.expander("Account metrics from order and cash history"):
        for m in perf["metrics"]:
            if m["name"] == "money_weighted_return_xirr" and not show_xirr:
                continue
            label = pv.label_for(m["name"])
            if m["status"] == "available":
                st.metric(label, _metric_text(m), help=m["basis"])
                if m.get("cross_check"):
                    st.caption(m["cross_check"])
                if m["name"] == "money_weighted_return_xirr":
                    for w in (perf["xirr"] or {}).get("warnings", []):
                        st.caption(w)
            elif m["status"] == "partial":
                shown = f" Unreconciled value: {_metric_text(m)}." if m.get("value") else ""
                st.warning(f"{label}: not reconciled.{shown} " + " ".join(m["blockers"]))
            else:
                st.info(f"{label}: unavailable. " + " ".join(m["blockers"][:2]))
    with st.expander("Data completeness"):
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Dataset": k["kind"],
                        "Records": k["records"],
                        "Earliest": (k["earliest_utc"] or "-")[:10],
                        "Latest": (k["latest_utc"] or "-")[:10],
                        "Checks passed": f"{sum(c['ok'] is True for c in k['checks'])}"
                        f"/{len(k['checks'])}",
                        "Types": ", ".join(f"{t} ({n})" for t, n in k["types"].items()),
                    }
                    for k in audit["kinds"]
                ]
            ),
            width="stretch",
            hide_index=True,
        )
        (st.success if audit["inception"]["ok"] else st.info)(audit["inception"]["detail"])
    if hold["positions"]:
        with st.expander("Position-by-position reconciliation"):
            st.dataframe(
                pd.DataFrame(hold["positions"])
                .assign(reasons=lambda f: f["reasons"].map(" ".join))
                .drop(columns=["reconstructed_cost", "broker_cost"]),
                width="stretch",
                hide_index=True,
            )
    if perf["external_flows"]:
        with st.expander("External cash flows used (investor-signed)"):
            st.dataframe(pd.DataFrame(perf["external_flows"]), width="stretch", hide_index=True)


st.title("MarketOps")
st.caption("Stock research and portfolio intelligence | Local-first | Read-only")
st.warning(
    "Research software, not investment advice. Held-out evaluation to date shows the ML signal "
    "has **no demonstrated predictive edge**; past performance is not indicative of the future."
)

ready, err = get("/health/ready")
if err:
    st.error(f"API or PostgreSQL not ready: {err}")
    st.stop()

tab_port, tab_act, tab_perf, tab_stock, tab_agent, tab_model, tab_bt = st.tabs(
    [
        "Portfolio",
        "Portfolio activity",
        "Historical performance",
        "Stock research",
        "AI research agent",
        "Model performance",
        "Backtests",
    ]
)

# ---------------------------------------------------------------- portfolio
with tab_port:
    source = st.radio(
        "Portfolio data", ["Synthetic example", "Trading 212 account"], horizontal=True
    )
    st.caption(
        "No orders can be placed from MarketOps. Account access needs a local read-only key."
    )
    if source == "Trading 212 account":
        st.text_input(
            "Local access token (MARKETOPS_AGENT_API_TOKEN)",
            type="password",
            key="tok_portfolio",
            help="Held only in this browser session's server memory; required for real data.",
        )
    if st.button("Load portfolio", type="primary"):
        endpoint = "/portfolio/example" if source == "Synthetic example" else "/portfolio/overview"
        data, e = get(
            endpoint, token_key="tok_portfolio" if endpoint.endswith("overview") else None
        )
        if e:
            st.session_state.pop("portfolio", None)
            st.session_state.portfolio_error = e
        else:
            st.session_state.pop("portfolio_error", None)
            st.session_state.portfolio = data
    if st.session_state.get("portfolio_error"):
        st.error(f"Could not load portfolio: {st.session_state.portfolio_error}")
        st.caption(
            "Check that keys are set in .env, the access token matches, and the API was "
            "restarted. Trading 212 allows about one request per second; wait and retry."
        )
    pf = st.session_state.get("portfolio")
    if pf:
        cur = pf["account_currency"]
        if pf["source"] == "trading212":
            env = (pf.get("broker_environment") or "unknown").upper()
            banner = st.error if env == "LIVE" else st.success
            banner(f"{env} TRADING 212 DATA: read-only snapshot. MarketOps cannot place orders.")
        else:
            st.warning("SYNTHETIC EXAMPLE: fabricated illustration values, not market data.")
        st.caption(
            f"Account currency: **{cur}**. Snapshot taken {pf['snapshot_utc']} (UTC); press "
            "Load portfolio to refresh. Values are broker-reported, not live quotes."
        )
        cash = pf.get("cash")
        a, b, c, d = st.columns(4)
        a.metric("Holdings value", f"{cur} {pf['holdings_value']:,.2f}")
        if cash:
            b.metric("Cash", f"{cur} {cash['total_cash']:,.2f}")
            incl = pf.get("portfolio_value_incl_cash")
            c.metric("Total incl. cash", f"{cur} {incl:,.2f}" if incl is not None else "Unverified")
        else:
            b.metric("Cash", "Unavailable")
            c.metric("Total incl. cash", "Unavailable")
        pct = pf.get("unrealised_pnl_pct")
        d.metric(
            "Unrealised P/L",
            f"{cur} {pf['unrealised_pnl']:,.2f}",
            f"{pct:+.2f}%" if pct is not None else None,
        )
        conc = pf["concentration"]
        for w in conc["warnings"]:
            st.warning(w)
        for w in pf["data_quality_warnings"]:
            st.info(w)
        df = pd.DataFrame(pf["holdings"])
        if df.empty:
            st.info("The account has no open positions.")
        else:
            left, right = st.columns(2)
            with left:
                st.subheader("Allocation by holding")
                st.bar_chart(df.set_index("ticker")["weight_pct"], y_label="% of holdings")
            with right:
                st.subheader("Exposure by instrument currency")
                ce = pd.DataFrame(pf["currency_exposure"])
                st.bar_chart(ce.set_index("key")["weight_pct"], y_label="% of holdings")
            e1, e2, e3, e4 = st.columns(4)
            e1.metric(
                "Largest position", f"{conc['largest_weight_pct']:.1f}%", conc["largest_ticker"]
            )
            e2.metric("Top 3 positions", f"{conc['top3_weight_pct']:.1f}%")
            e3.metric("Effective no. of holdings", f"{conc['effective_number_of_holdings']:.1f}")
            e4.metric("Herfindahl index", f"{conc['herfindahl_index']:.3f}")
            st.subheader("Holdings")
            table = df.rename(
                columns={
                    "ticker": "Ticker",
                    "name": "Name",
                    "quantity": "Qty",
                    "instrument_currency": "Instr. ccy",
                    "current_value": f"Value ({cur})",
                    "total_cost": f"Cost ({cur})",
                    "unrealised_pnl": f"P/L ({cur})",
                    "unrealised_pnl_pct": "P/L %",
                    "fx_impact": f"FX impact ({cur})",
                    "weight_pct": "Weight %",
                }
            )[
                [
                    "Ticker",
                    "Name",
                    "Qty",
                    "Instr. ccy",
                    f"Value ({cur})",
                    f"Cost ({cur})",
                    f"P/L ({cur})",
                    "P/L %",
                    f"FX impact ({cur})",
                    "Weight %",
                ]
            ]
            st.dataframe(table, width="stretch", hide_index=True)
        if pf["cash_pct_of_portfolio"] is not None:
            st.caption(f"Cash is {pf['cash_pct_of_portfolio']:.1f}% of holdings plus cash.")
        if pf.get("broker_realised_pnl_all_time") is not None:
            st.caption(
                f"Broker-reported all-time realised P/L: {cur} "
                f"{pf['broker_realised_pnl_all_time']:,.2f} (not a return figure)."
            )
        recon = pf.get("reconciliation") or []
        if recon:
            with st.expander("Reconciliation: broker summary vs computed from positions"):
                st.dataframe(
                    pd.DataFrame(recon).rename(
                        columns={
                            "metric": "Metric",
                            "broker_reported": f"Broker summary ({cur})",
                            "computed": f"Computed from positions ({cur})",
                            "difference": "Difference",
                            "status": "Status",
                        }
                    ),
                    width="stretch",
                    hide_index=True,
                )
                st.caption(
                    "Summary and positions are separate requests, so tiny differences can occur. "
                    "Allocation uses positions; the summary is shown for audit only."
                )
        st.info(f"Historical performance: not available. {pf['performance']['reason']}")
        st.caption(pf["note"])
    elif not st.session_state.get("portfolio_error"):
        st.info("Press Load portfolio. No account is accessed automatically.")

# ---------------------------------------------------------------- portfolio activity
_STATE_TEXT = {
    "never_synced": "Not imported",
    "running": "Sync running",
    "complete": "Complete",
    "stale": "Complete, last update failed",
    "partial": "Partial: may be missing records",
}


def _totals(rows: list[dict[str, Any]]) -> str:
    return " + ".join(f"{r['currency']} {r['amount']:,.2f}" for r in rows) if rows else "none"


with tab_act:
    st.subheader("Portfolio activity")
    st.caption(
        "Deposits, withdrawals, dividends and trades imported read-only from Trading 212 and "
        "stored in your local PostgreSQL. Ordinary page loads never download anything: only the "
        "'Sync account history' button contacts the broker."
    )
    st.text_input(
        "Local access token (MARKETOPS_AGENT_API_TOKEN)",
        type="password",
        key="tok_history",
        help="Held only in this browser session's server memory. Required for every call here.",
    )
    tk = "tok_history" if st.session_state.get("tok_history") else "tok_portfolio"

    with st.expander("Sync account history", expanded=False):
        st.write(
            "Downloads your order, cash-transaction and dividend history with read-only GET "
            "requests. Trading 212 limits history endpoints to 20 requests per minute, so a long "
            "history takes several minutes. Re-running is safe: stored records are never "
            "duplicated."
        )
        confirm = st.checkbox("I want to download my account history now", key="act_confirm")
        force_full = st.checkbox(
            "Force a complete re-pass (instead of only new records)", key="act_full"
        )
        if st.button("Sync account history", type="primary", disabled=not confirm):
            data, err = post("/history/sync", {"full": force_full}, timeout=60.0, token_key=tk)
            if err:
                st.session_state.pop("act_sync_msg", None)
                st.session_state.act_sync_error = err
            else:
                st.session_state.pop("act_sync_error", None)
                st.session_state.act_sync_msg = (
                    f"Sync started on the {data['environment'].upper()} account "
                    f"({data['account_currency']}). "
                    "Press 'Load saved history' to refresh progress."
                )
        if st.session_state.get("act_sync_error"):
            st.error(f"Could not start the sync: {st.session_state.act_sync_error}")
        if st.session_state.get("act_sync_msg"):
            st.success(st.session_state.act_sync_msg)

    with st.form("act_filter"):
        c1, c2 = st.columns(2)
        d_from = c1.date_input("From (optional)", value=None, key="act_from")
        d_to = c2.date_input("To (optional)", value=None, key="act_to")
        submitted = st.form_submit_button("Load saved history")
    if submitted:
        params = {
            k: v.isoformat() for k, v in (("date_from", d_from), ("date_to", d_to)) if v is not None
        }
        out: dict[str, Any] = {"params": params}
        errors: list[str] = []
        for name in ("status", "summary", "readiness", "transactions", "orders", "dividends"):
            data, err = get(
                f"/history/{name}",
                token_key=tk,
                **(params if name not in ("status", "readiness") else {}),
            )
            out[name] = data
            if err and not (name in ("summary", "readiness") and "No account history" in err):
                errors.append(f"{name}: {err}")
        st.session_state.act = out
        st.session_state.act_errors = errors

    for err in st.session_state.get("act_errors", []):
        st.error(f"Could not load {err}")
    act = st.session_state.get("act")
    status_body = act.get("status") if act else None
    if status_body:
        env_name = (status_body.get("environment") or "unknown").upper()
        (st.error if env_name == "LIVE" else st.success)(
            f"{env_name} TRADING 212 HISTORY: stored read-only copy. MarketOps cannot place orders."
        )
        if status_body.get("sync_running"):
            st.info("A sync is running. Press 'Load saved history' to refresh.")
        cov = pd.DataFrame(status_body["kinds"])
        st.markdown("**Data completeness**")
        shown = cov.assign(
            state=cov["state"].map(_STATE_TEXT),
            earliest=cov["earliest_record_utc"].fillna("-").str[:10],
            latest=cov["latest_record_utc"].fillna("-").str[:10],
            last_ok=cov["last_success_utc"].fillna("never").str[:19],
        )[["kind", "state", "stored_records", "earliest", "latest", "last_ok"]].rename(
            columns={
                "kind": "Dataset",
                "state": "State",
                "stored_records": "Records",
                "earliest": "Earliest record",
                "latest": "Latest record",
                "last_ok": "Last successful sync (UTC)",
            }
        )
        st.dataframe(shown, width="stretch", hide_index=True)
        for k in status_body["kinds"]:
            for issue in k["issues"]:
                st.warning(f"{k['kind']}: {issue}")
        st.caption(
            "Earliest record shows how far back the API returned data. MarketOps never infers "
            "events before it or between records."
        )
        if all(k["state"] == "never_synced" for k in status_body["kinds"]):
            st.info("No account history has been imported yet. Use 'Sync account history' above.")

    summ = act.get("summary") if act else None
    if summ:
        st.markdown("**Activity overview** (selected dates; per currency, no conversion)")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Deposits", _totals(summ["deposits"]))
        m2.metric("Withdrawals", _totals(summ["withdrawals"]))
        m3.metric("Dividends received", _totals(summ["dividends"]))
        m4.metric("Cash fees", _totals(summ["cash_fees"]))
        n1, n2, n3, n4 = st.columns(4)
        n1.metric("Buys (filled)", summ["buy_count"], _totals(summ["buy_value"]), delta_color="off")
        n2.metric(
            "Sells (filled)", summ["sell_count"], _totals(summ["sell_value"]), delta_color="off"
        )
        n3.metric("Interest on cash", _totals(summ["interest"]))
        n4.metric("Order fees/taxes", _totals(summ["order_fees"]))
        st.caption(
            "Buys and sells are trades inside the account, not deposits or withdrawals. "
            f"{summ['transfers_count']} TRANSFER record(s) are listed but not counted as flows; "
            f"{summ['corporate_action_count']} corporate-action fill(s); "
            f"{summ['unfilled_order_count']} order(s) without a fill."
        )
        if summ["excluded_inconsistent"]:
            st.warning(
                f"{summ['excluded_inconsistent']} inconsistent record(s) are excluded from totals."
            )
        if summ["dividends_by_month"]:
            dm = pd.DataFrame(summ["dividends_by_month"])
            st.markdown("**Dividends by month**")
            st.bar_chart(
                dm.pivot_table(index="month", columns="currency", values="amount", aggfunc="sum")
            )

    if act and status_body:
        t_tx, t_ord, t_div, t_perf = st.tabs(
            ["Transactions", "Buy/sell activity", "Dividend history", "Performance metrics"]
        )
        for tab, name in ((t_tx, "transactions"), (t_ord, "orders"), (t_div, "dividends")):
            with tab:
                page = act.get(name)
                if page is None:
                    st.info("Not loaded.")
                elif not page["items"]:
                    st.info("No records for this selection.")
                else:
                    st.caption(f"Showing {len(page['items'])} of {page['total']} (newest first).")
                    frame = pd.DataFrame(page["items"])
                    st.dataframe(
                        frame.drop(columns=["taxes"], errors="ignore"),
                        width="stretch",
                        hide_index=True,
                    )
        with t_perf:
            ready = act.get("readiness")
            if not ready:
                st.info("Import history to see which metrics are verified.")
            else:
                st.caption(
                    "Only metrics whose inputs are verified are shown. Return percentages, XIRR, "
                    "time-weighted return, an equity curve and benchmark comparison are not "
                    "calculated: transaction history alone does not establish portfolio value over "
                    "time."
                )
                for m in ready:
                    label = m["metric"].replace("_", " ")
                    if m["status"] == "available":
                        st.success(f"{label}: {m['value']}")
                    elif m["status"] == "partial":
                        st.warning(f"{label}: unverified. " + " ".join(m["blockers"]))
                    else:
                        st.info(f"{label}: unavailable. " + " ".join(m["blockers"][:1]))

# ---------------------------------------------------------------- historical performance
with tab_perf:
    st.subheader("Historical performance")
    st.caption(
        "Daily portfolio value and returns rebuilt from your stored order, cash and dividend "
        "history, valued at **historical closing prices**, and checked against a read-only "
        "**live Trading 212 snapshot**. Deposits are never counted as profit; a figure is shown "
        "as reconciled only when its inputs match the live account."
    )
    st.text_input(
        "Local access token (MARKETOPS_AGENT_API_TOKEN)",
        type="password",
        key="tok_perf",
        help="Held only in this browser session's server memory.",
    )
    tkp = "tok_perf" if st.session_state.get("tok_perf") else "tok_history"
    b1, b2, b3, b4 = st.columns(4)
    if b1.button("Calculate historical valuation", type="primary"):
        st.session_state.val, st.session_state.val_err = get(
            "/performance/valuation", token_key=tkp
        )
    if b2.button("Calculate historical performance"):
        st.session_state.perf, st.session_state.perf_err = get(
            "/performance/historical", token_key=tkp
        )
    if b4.button("Load synthetic demo"):
        st.session_state.val, st.session_state.val_err = get("/performance/demo")
        st.session_state.perf = st.session_state.perf_err = None
    if b3.button("Download prices, FX and S&P 500 data"):
        st.session_state.val_refresh, st.session_state.val_refresh_err = post(
            "/performance/valuation/refresh", {}, token_key=tkp
        )
    if st.session_state.get("val_refresh_err"):
        st.error(f"Download failed: {st.session_state.val_refresh_err}")
    elif st.session_state.get("val_refresh"):
        st.caption(
            "Download results: "
            + "; ".join(f"{k}: {v}" for k, v in st.session_state.val_refresh.items())
        )
    if st.session_state.get("val_err"):
        st.error(f"Could not calculate: {st.session_state.val_err}")
    if st.session_state.get("perf_err"):
        st.error(f"Could not calculate: {st.session_state.perf_err}")
    val, perf = st.session_state.get("val"), st.session_state.get("perf")
    if val:
        _render_valuation(val, perf)
    if perf:
        st.divider()
        _render_snapshot(perf, show_xirr=not val)
    if not val and not perf:
        st.info(
            "Press “Calculate historical valuation” for portfolio value, returns and charts, and "
            "“Calculate historical performance” for the live-account reconciliation and XIRR."
        )
    elif perf and not val:
        st.info("Press “Calculate historical valuation” for portfolio value, returns and charts.")


# ---------------------------------------------------------------- stock research
with tab_stock:
    meta, e = get("/model/metadata")
    if e:
        st.info(f"No trained model available: {e}")
    else:
        sym = st.selectbox("Stock", meta["symbols"])
        days = st.slider("History (trading days)", 60, 2500, 500, step=20)
        hist, e = get(f"/market/{sym}/history", days=days)
        if e:
            st.error(e)
        else:
            h = pd.DataFrame(hist["rows"]).assign(date=lambda d: pd.to_datetime(d["date"]))
            h = h.set_index("date")
            st.subheader(f"{sym} adjusted close")
            st.line_chart(h["adj_close"])
            st.subheader("Volume")
            st.bar_chart(h["volume"])
            st.caption(f"Source: {hist['provider']}. {hist['price_convention']}. {hist['terms']}")
        if st.button("Compute ML signal (latest bar)"):
            an, e = get(f"/analysis/{sym}")
            if e:
                st.error(e)
            else:
                st.session_state.analysis = an
        an = st.session_state.get("analysis")
        if an and an["symbol"] == sym:
            a, b, c = st.columns(3)
            a.metric("P(up, next 5 sessions)", f"{an['probability_up']:.1%}")
            b.metric("Decision threshold", f"{an['threshold']:.1%}")
            c.metric("As-of date", an["as_of_date"])
            for w in an["warnings"]:
                st.warning(w)
            st.caption(
                f"Model {an['model_version']}. See the Model performance tab for held-out "
                "evidence; small deviations from the base rate are not evidence of edge."
            )
            st.caption(an["disclaimer"])
            st.dataframe(pd.Series(an["features"], name="value"), width="stretch")

# ---------------------------------------------------------------- research agent
with tab_agent:
    st.caption(
        "A local Ollama model calls read-only tools and writes a summary. Claims are checked "
        "against retrieved evidence; unsupported ones are shown as rejected. It cannot trade."
    )
    stat, _ = get("/agent/status")
    if stat:
        llm = stat["llm"]
        if llm["ollama"] == "up" and llm["model_installed"]:
            st.caption(f"Local model: {llm['model']} (ready)")
        else:
            st.warning(
                f"Local LLM unavailable ({llm.get('detail') or llm['ollama']}). Requests will "
                "return a clearly labelled template summary built without the LLM."
            )
    q = st.text_area(
        "Research question",
        "Summarise AAPL: recent price action, what the ML model says, and how much to trust it.",
        max_chars=1000,
    )
    src = st.radio(
        "Portfolio context",
        ["none", "example", "trading212"],
        horizontal=True,
        format_func=lambda s: {
            "none": "None (public-market research)",
            "example": "Synthetic demonstration portfolio",
            "trading212": "My Trading 212 account (read-only; must be enabled locally)",
        }[s],
    )
    if src == "trading212":
        st.text_input(
            "Local access token (MARKETOPS_AGENT_API_TOKEN)",
            type="password",
            key="tok_agent",
            help="Required for real account data.",
        )
    if st.button("Run research", type="primary"):
        with st.spinner("Researching with the local model (can take a few minutes on CPU)..."):
            rep, e = post(
                "/agent/research",
                {"question": q, "portfolio_source": src},
                token_key="tok_agent" if src == "trading212" else None,
            )
        if e:
            st.error(e)
            st.session_state.pop("agent_report", None)
        else:
            st.session_state.agent_report = rep
    rep = st.session_state.get("agent_report")
    if rep:
        st.error(rep["disclaimer"])
        if rep["mode"] != "llm":
            st.warning(f"Template summary, not model-written (LLM status: {rep['llm_status']}).")
        if rep["portfolio_source"] == "example":
            st.warning("SYNTHETIC DEMONSTRATION portfolio, not a real account.")
        for w in rep["warnings"]:
            st.warning(w)
        for title, key in (
            ("Verified historical facts", "verified_facts"),
            ("Model-generated estimates (no demonstrated edge)", "model_estimates"),
            ("Analytical interpretation", "interpretation"),
        ):
            st.subheader(title)
            if not rep[key]:
                st.caption("None passed verification.")
            for c in rep[key]:
                st.markdown(f"- {c['text']}  `{', '.join(c['evidence']) or 'no evidence'}`")
        st.subheader("Missing or uncertain")
        for m in rep["missing_or_uncertain"] or ["Nothing flagged."]:
            st.markdown(f"- {m}")
        st.subheader("Model limitations")
        for lim in rep["limitations"]:
            st.markdown(f"- {lim}")
        if rep["rejected_claims"]:
            with st.expander(f"Rejected claims ({len(rep['rejected_claims'])})"):
                st.dataframe(pd.DataFrame(rep["rejected_claims"]), hide_index=True)
        with st.expander(f"Evidence ({len(rep['evidence'])})"):
            ev = pd.DataFrame(rep["evidence"])[["id", "tool", "kind", "source", "retrieved_utc"]]
            st.dataframe(ev, hide_index=True)
        with st.expander("Tool calls and audit"):
            st.caption(
                f"Run {rep['run_id']} | {rep['llm_model']} | {rep['iterations']} iterations | "
                f"{rep['tool_calls']} tool calls | {rep['duration_ms'] / 1000:.1f}s"
            )
            st.dataframe(pd.DataFrame(rep["tool_trace"]), hide_index=True)

# ---------------------------------------------------------------- model performance
with tab_model:
    ev, e = get("/model/evaluation")
    meta, _ = get("/model/metadata")
    if e:
        st.info(f"No evaluation available: {e}")
    else:
        st.subheader(f"Model {ev['version']} | selected: {meta['selected_model']}")
        st.caption(
            f"Target: {meta['target']}. Development to {meta['dev_end']} (walk-forward CV); "
            f"held-out test from {meta['test_start']}, evaluated once."
        )
        cv = pd.DataFrame(ev["walk_forward_cv"]).T
        st.markdown("**Walk-forward CV (development only)**")
        st.dataframe(cv, width="stretch")
        t = ev["test"]
        rows = {
            "Selected model": t["model"],
            "Always-up baseline": t["baseline_always_up"],
            "Momentum-21d baseline": t["baseline_momentum_21d"],
        }
        keys = [
            "roc_auc",
            "pr_auc",
            "brier",
            "log_loss",
            "precision",
            "recall",
            "f1",
            "balanced_accuracy",
        ]
        st.markdown("**Held-out test**")
        st.dataframe(
            pd.DataFrame({k: {m: v[m] for m in keys} for k, v in rows.items()}).T, width="stretch"
        )
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Confusion matrix (selected, test)**")
            cm = t["model"]["confusion"]
            st.dataframe(
                pd.DataFrame(
                    [[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]],
                    index=["actual down", "actual up"],
                    columns=["pred down", "pred up"],
                )
            )
        with c2:
            st.markdown("**Calibration (test, deciles)**")
            cal = pd.DataFrame(t["model"]["calibration"]).set_index("mean_pred")["observed"]
            st.line_chart(cal)
        lat = meta["latency"]
        st.caption(
            f"Single-row inference p50 {lat['single_row_ms_p50']:.2f} ms, "
            f"p95 {lat['single_row_ms_p95']:.2f} ms (CPU, in-process)."
        )
        for cav in ev["caveats"]:
            st.caption(f"Caveat: {cav}")

        shap_data, e = get("/model/explain")
        if not e:
            st.markdown("**Feature importance (mean |SHAP|, log-odds, held-out sample)**")
            st.bar_chart(pd.Series(shap_data["mean_abs_shap_logodds"]))
        mon, e = get("/monitoring")
        if not e:
            st.markdown("**Monitoring**")
            if mon["drift"]:
                d = mon["drift"]
                fp = pd.DataFrame(d["feature_psi"]).T
                st.caption(
                    f"Drift: last 60 sessions ({d['recent_window'][0]} to {d['recent_window'][1]}) "
                    f"vs development data. PSI >= {d['thresholds']['warn']} warn, "
                    f">= {d['thresholds']['alert']} alert. Prediction PSI "
                    f"{d['prediction_psi']['psi']:.3f} ({d['prediction_psi']['level']})."
                )
                st.dataframe(fp, width="stretch")
            st.dataframe(pd.DataFrame(mon["data_quality"]).T, width="stretch")

# ---------------------------------------------------------------- backtests
with tab_bt:
    bt, e = get("/backtest")
    if e:
        st.info(f"No backtest available: {e}")
    else:
        st.caption(
            f"Signals at close, fills at next open. {bt['cost_bps_one_way']:.0f} bps one-way "
            f"cost assumption; rebalance every {bt['rebalance_every_sessions']} sessions."
        )
        meta_bt, _ = get("/model/metadata")
        test_from = meta_bt["test_start"][:4] if meta_bt else "test start"
        split = st.radio(
            "Period",
            ["test", "dev_oof"],
            horizontal=True,
            format_func=lambda s: (
                f"Held-out test ({test_from}+)"
                if s == "test"
                else "Development (walk-forward out-of-fold)"
            ),
        )
        st.caption(f"Backtest of model {bt['model_version']} (threshold {bt['threshold']:.3f}).")
        m = pd.DataFrame(bt[split]).T
        st.dataframe(m.rename(columns=str), width="stretch")
        eq = pd.DataFrame(bt["equity_curves"][split])
        eq.index = pd.to_datetime(eq.index)
        st.line_chart(eq)
        for a in bt["assumptions"]:
            st.caption(a)
        st.caption("Model strategy parameters were not tuned on the test period.")
