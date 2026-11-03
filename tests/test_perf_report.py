"""Historical performance report, API and dashboard on synthetic accounts (no real data).

Reference account (GBP): deposits 200 + 500, withdrawal 50, fee 1, dividends 4.0 + 3.2, two buys of
100 (2.5 shares each), one sale of 120 (2.5 shares, broker P/L 20). Cash therefore ends at
700 - 50 - 1 + 7.2 - 200 + 120 = 576.2 and 2.5 shares remain at cost 100.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import httpx
import pytest

from marketops.history import read
from marketops.performance import report as report_mod
from marketops.performance.report import HistoricalPerformance, build_report
from marketops.performance.xirr import XirrResult
from marketops.portfolio.analytics import PortfolioOverview, summarise_positions
from tests.history_fixtures import (
    HDR,
    KEY,
    SECRET,
    TOKEN,
    Env,
    div,
    order,
    standard_pages,
    tx,
)

pytestmark = pytest.mark.filterwarnings("ignore:Dialect sqlite.*Decimal")


def snapshot(
    qty: float = 2.5,
    cash: float = 576.2,
    realised: float = 20.0,
    positions: bool = True,
    ticker: str = "AAA_US_EQ",
) -> PortfolioOverview:
    rows = (
        [
            {
                "instrument": {"ticker": ticker, "name": "Synthetic AAA", "currency": "USD"},
                "quantity": qty,
                "walletImpact": {
                    "currency": "GBP",
                    "currentValue": 130,
                    "totalCost": 100,
                    "unrealizedProfitLoss": 30,
                },
            }
        ]
        if positions
        else []
    )
    value = 130.0 if positions else 0.0
    summary = {
        "currency": "GBP",
        "totalValue": value + cash,
        "cash": {"availableToTrade": cash, "inPies": 0, "reservedForOrders": 0},
        "investments": {
            "currentValue": value,
            "totalCost": 100 if positions else 0,
            "realizedProfitLoss": realised,
            "unrealizedProfitLoss": 30 if positions else 0,
        },
    }
    out = summarise_positions(rows, source="trading212", summary=summary)
    out.broker_environment = "demo"
    return out


def run(env: Env, snap: PortfolioOverview | None, pages: Any = None) -> HistoricalPerformance:
    env.source.pages = standard_pages() if pages is None else pages
    env.sync(full=True)
    key = read.latest_account_key(env.engine, "demo")
    assert key
    cov = read.coverage(env.engine, "demo", key, False)
    return build_report(env.engine, "demo", key, cov, snap)


def metric(rep: HistoricalPerformance, name: str) -> Any:
    return next(m for m in rep.metrics if m.name == name)


# ------------------------------------------------------------------ the reconciled account
def test_reconciled_account_is_verified_and_every_figure_is_exact(env: Env) -> None:
    rep = run(env, snapshot())
    assert rep.overall_status == "verified" and rep.holdings.status == "reconciled"
    assert rep.cash_ledger.status == "match" and rep.cash_ledger.difference == pytest.approx(0)
    assert rep.data_audit.inception.ok is True
    assert metric(rep, "total_external_deposits").value == {"amount": 700.0, "count": 2.0}
    assert metric(rep, "total_external_withdrawals").value == {"amount": 50.0, "count": 1.0}
    assert metric(rep, "net_invested_capital").value == {"amount": 650.0}
    assert metric(rep, "dividends_received").value == {"amount": 7.2, "count": 2.0}
    assert metric(rep, "account_fees").value == {"cash_fees": 1.0}
    realised = metric(rep, "realised_pnl")
    assert realised.status == "available"
    assert realised.value == {
        "broker_reported_sum_of_fills": 20.0,
        "reconstructed_average_cost": pytest.approx(20.0),
    }
    assert metric(rep, "unrealised_pnl").value == {"amount": 30.0}
    gain = metric(rep, "net_gain_on_invested_capital")
    # 706.2 + 50 - 700 = realised 20 + unrealised 30 + dividends 7.2 - fee 1
    assert gain.value is not None and gain.value["gain"] == pytest.approx(56.2)
    assert all(c.ok for k in rep.data_audit.kinds for c in k.checks)


def test_buys_sells_dividends_and_interest_are_never_external_flows(env: Env) -> None:
    rep = run(env, snapshot())
    kinds = [f.kind for f in rep.external_flows]
    assert kinds == ["deposit", "deposit", "withdrawal", "terminal_value"]
    amounts = [f.amount for f in rep.external_flows]
    assert amounts[:3] == [-200.0, -500.0, 50.0]  # investor-signed: paid in negative
    assert [p.net_invested for p in rep.net_invested_series] == [200.0, 700.0, 650.0]


def test_xirr_matches_an_independent_solution(env: Env) -> None:
    rep = run(env, snapshot())
    x = rep.xirr
    assert x is not None and x.status == "ok" and x.rate is not None
    from datetime import datetime

    flows = [(datetime.fromisoformat(f.date), f.amount) for f in rep.external_flows]
    t0 = flows[0][0]
    npv = sum(a * (1 + x.rate) ** (-(d - t0).total_seconds() / 86400 / 365) for d, a in flows)
    assert abs(npv) < 1e-6 and metric(rep, "money_weighted_return_xirr").status == "available"


def test_return_metrics_that_need_valuation_history_are_unavailable_with_reasons(env: Env) -> None:
    rep = run(env, snapshot())
    for name in (
        "time_weighted_return",
        "portfolio_equity_curve",
        "total_return_pct",
        "benchmark_comparison",
    ):
        m = metric(rep, name)
        assert m.status == "unavailable" and m.value is None and m.blockers


def test_deposits_alone_are_not_profit(env: Env) -> None:
    pages: Any = {
        "orders": [[]],
        "transactions": [[tx("T1", "DEPOSIT", 1000.0, day=3)]],
        "dividends": [[]],
    }
    rep = run(env, snapshot(positions=False, cash=1000.0, realised=0.0), pages)
    gain = metric(rep, "net_gain_on_invested_capital")
    assert gain.value is not None and gain.value["gain"] == pytest.approx(0.0)
    assert rep.xirr is not None and rep.xirr.rate == pytest.approx(0.0, abs=1e-9)
    assert metric(rep, "net_invested_capital").value == {"amount": 1000.0}


# ------------------------------------------------------------------ classification
def test_dividend_interest_and_distribution_types_are_kept_apart(env: Env) -> None:
    pages = standard_pages()
    pages["transactions"][0] += [
        tx("I1", "INTEREST_ON_FREE_CASH", 2.0, day=6),
        tx("I2", "LENDING_INTEREST", 1.0, day=7),
    ]
    pages["dividends"][0] += [
        div("D3", 0.5, kind="INTEREST", day=21),
        div("D4", 1.0, kind="CAPITAL_GAIN", day=22),
    ]
    rep = run(env, snapshot(cash=576.2 + 3.0 + 0.5 + 1.0), pages)
    assert metric(rep, "dividends_received").value == {"amount": 7.2, "count": 2.0}
    interest = metric(rep, "interest_received").value
    assert interest == {"cash_interest": 3.0, "dividend_interest": 0.5}
    assert rep.cash_ledger.status == "match"  # the capital distribution is cash, not dividend
    assert rep.cash_ledger.components["dividends_and_distributions"] == pytest.approx(8.7)


def test_transfers_are_not_flows_and_block_the_return(env: Env) -> None:
    pages = standard_pages()
    pages["transactions"][0].append(tx("X1", "TRANSFER", 25.0, day=8))
    rep = run(env, snapshot(), pages)
    assert metric(rep, "total_external_deposits").value == {"amount": 700.0, "count": 2.0}
    assert metric(rep, "total_external_deposits").status == "partial"
    assert any("TRANSFER" in b for b in metric(rep, "net_invested_capital").blockers)
    assert rep.xirr is None and metric(rep, "money_weighted_return_xirr").status == "unavailable"
    assert rep.overall_status != "verified"


# ------------------------------------------------------------------ incomplete or inconsistent data
def test_incomplete_history_blocks_every_dependent_metric(env: Env) -> None:
    from marketops.broker.t212 import BrokerError

    env.source.fail_at[("transactions", 1)] = BrokerError("down", "unavailable")
    rep = run(env, snapshot())
    assert metric(rep, "net_invested_capital").status == "partial"
    assert any("not complete" in b for b in metric(rep, "net_invested_capital").blockers)
    assert metric(rep, "net_gain_on_invested_capital").status == "unavailable"
    assert rep.xirr is None
    assert rep.overall_status != "verified"
    assert any("transactions" in i for i in rep.data_audit.issues)


def test_non_account_currency_flow_is_not_converted(env: Env) -> None:
    pages = standard_pages()
    pages["transactions"][0].append(tx("U1", "DEPOSIT", 100.0, day=9, ccy="USD"))
    rep = run(env, snapshot(), pages)
    assert metric(rep, "total_external_deposits").status == "partial"
    assert any("currency" in b for b in metric(rep, "total_external_deposits").blockers)
    assert metric(rep, "total_external_deposits").value == {"amount": 700.0, "count": 2.0}
    assert rep.xirr is None


def test_missing_fx_rate_is_flagged_on_the_reconstructed_position(env: Env) -> None:
    pages = standard_pages()
    bad = copy.deepcopy(pages["orders"][1][0])
    bad["fill"]["walletImpact"].pop("fxRate")
    pages["orders"][1][0] = bad
    rep = run(env, snapshot(), pages)
    assert "missing_fx_rate" in rep.reconstructed_positions[0].flags
    assert rep.holdings.status == "reconciled"  # quantities do not depend on FX


def test_trade_in_a_foreign_wallet_currency_is_excluded_and_blocks_verification(env: Env) -> None:
    pages = standard_pages()
    pages["orders"][2].append(order(9, "BUY", value=-40.0, day=21, ccy="EUR"))
    rep = run(env, snapshot(), pages)
    assert metric(rep, "realised_pnl").status == "partial"
    assert rep.xirr is None and rep.overall_status != "verified"
    assert any("inconsistent" in i for i in rep.data_audit.issues)


# ------------------------------------------------------------------ reconciliation failures
def test_position_mismatch_blocks_returns_and_is_never_repaired(env: Env) -> None:
    rep = run(env, snapshot(qty=3.0))
    pos = rep.holdings.positions[0]
    assert rep.holdings.status == "mismatch" and pos.status == "quantity_mismatch"
    assert pos.reconstructed_quantity == 2.5 and pos.broker_quantity == 3.0
    assert pos.quantity_difference == pytest.approx(-0.5) and pos.reasons
    assert rep.xirr is None
    assert metric(rep, "net_gain_on_invested_capital").status == "unavailable"
    assert metric(rep, "unrealised_pnl").status == "partial"
    assert rep.overall_status == "unverified"
    assert rep.reconstructed_positions[0].quantity == 2.5  # not overwritten by the broker figure


def test_corporate_action_gap_is_explained_and_not_papered_over(env: Env) -> None:
    pages = standard_pages()
    pages["orders"][2].append(order(8, "BUY", value=0.0, day=21, fill_type="STOCK_SPLIT"))
    rep = run(env, snapshot(qty=5.0), pages)
    pos = rep.holdings.positions[0]
    assert pos.status == "quantity_mismatch"
    assert any("corporate action" in r for r in pos.reasons)
    assert rep.xirr is None and rep.overall_status == "unverified"


def test_missing_deposit_shows_in_the_cash_ledger_and_denies_inception(env: Env) -> None:
    pages = standard_pages()
    pages["transactions"][1] = [tx("T2", "DEPOSIT", 500.0, day=10)]  # the 200 deposit is missing
    rep = run(env, snapshot(), pages)
    assert rep.holdings.status == "reconciled"
    assert rep.cash_ledger.status == "mismatch"
    assert rep.cash_ledger.difference == pytest.approx(-200.0)
    assert rep.data_audit.inception.ok is False
    assert metric(rep, "total_external_deposits").status == "partial"
    assert rep.xirr is None


def test_without_a_live_snapshot_nothing_is_verified(env: Env) -> None:
    rep = run(env, None)
    assert not rep.snapshot_available and rep.holdings.status == "not_checked"
    assert rep.cash_ledger.status == "not_checked" and rep.xirr is None
    assert metric(rep, "unrealised_pnl").status == "unavailable"
    assert metric(rep, "total_external_deposits").status == "partial"
    assert metric(rep, "money_weighted_return_xirr").status == "unavailable"
    assert rep.overall_status == "unverified"


def test_ambiguous_xirr_is_never_shown_as_a_number(
    env: Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        report_mod,
        "xirr",
        lambda flows: XirrResult(status="ambiguous", roots=[0.1, 0.2], reason="2 rates"),
    )
    m = metric(run(env, snapshot()), "money_weighted_return_xirr")
    assert m.status == "partial" and m.value is None
    assert any("ambiguous" in b for b in m.blockers)


def test_records_after_the_snapshot_cannot_leak_into_it(env: Env) -> None:
    from datetime import UTC, datetime

    early = snapshot().model_copy(update={"snapshot_utc": datetime(2024, 3, 12, tzinfo=UTC)})
    rep = run(env, early)
    kinds = {k.kind: k for k in rep.data_audit.kinds}
    assert kinds["orders"].records == 1  # only the unfilled order on 10 March
    assert kinds["dividends"].records == 0
    assert rep.reconstructed_positions == []  # buys on 18/19 March are in the future
    assert rep.holdings.status == "mismatch"  # so nothing is reported as verified
    assert rep.xirr is None
    assert any("excluded" in c.detail for c in kinds["orders"].checks)


# ------------------------------------------------------------------ API
@pytest.fixture
def live(env: Env, monkeypatch: pytest.MonkeyPatch) -> Env:
    monkeypatch.setattr("api.routes.performance._live_snapshot", lambda: (snapshot(), None))
    return env


def test_performance_route_needs_the_token(live: Env) -> None:
    assert live.client.get("/performance/historical").status_code == 401
    wrong = {"X-MarketOps-Token": "wrong-token-value-123"}
    assert live.client.get("/performance/historical", headers=wrong).status_code == 401


def test_performance_route_is_denied_when_no_token_is_configured(
    live: Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    from marketops.config import get_settings

    monkeypatch.delenv("MARKETOPS_AGENT_API_TOKEN")
    get_settings.cache_clear()
    assert live.client.get("/performance/historical", headers=HDR).status_code == 403


def test_performance_route_without_history_is_404(live: Env) -> None:
    assert live.client.get("/performance/historical", headers=HDR).status_code == 404


def test_performance_route_returns_the_verified_report_without_secrets(live: Env) -> None:
    live.sync(full=True)
    r = live.client.get("/performance/historical", headers=HDR)
    assert r.status_code == 200
    body = r.json()
    assert body["overall_status"] == "verified" and body["snapshot_available"] is True
    assert body["environment"] == "demo"
    for secret in (KEY, SECRET, TOKEN):
        assert secret not in r.text


def test_performance_route_reports_snapshot_failure_instead_of_guessing(
    env: Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "api.routes.performance._live_snapshot", lambda: (None, "Live snapshot unavailable (502)")
    )
    env.sync(full=True)
    body = env.client.get("/performance/historical", headers=HDR).json()
    assert body["overall_status"] == "unverified" and body["xirr"] is None
    assert any("Live snapshot unavailable" in n for n in body["notes"])


def test_snapshot_from_another_environment_is_rejected(
    env: Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    other = snapshot()
    other.broker_environment = "live"
    monkeypatch.setattr("api.routes.performance._live_snapshot", lambda: (other, None))
    env.sync(full=True)
    body = env.client.get("/performance/historical", headers=HDR).json()
    assert body["snapshot_available"] is False
    assert any("different environment" in n for n in body["notes"])


# ------------------------------------------------------------------ dashboard
DASHBOARD = str(Path(__file__).resolve().parents[1] / "app" / "dashboard.py")


def _app(env: Env, monkeypatch: pytest.MonkeyPatch) -> Any:
    from streamlit.testing.v1 import AppTest

    api = "http://localhost:8001"

    def fake_get(url: str, **kw: Any) -> Any:
        if "/health" in url:
            return type("R", (), {"status_code": 200, "json": lambda self: {"status": "ready"}})()
        return env.client.get(
            url.removeprefix(api), params=kw.get("params"), headers=kw.get("headers")
        )

    monkeypatch.setattr(httpx, "get", fake_get)
    return AppTest.from_file(DASHBOARD, default_timeout=60).run()


def _calculate(at: Any, token: str | None = TOKEN) -> None:
    if token:
        at.text_input(key="tok_perf").set_value(token).run()
    next(b for b in at.button if b.label == "Calculate historical performance").click()
    at.run()


def test_dashboard_renders_verified_metrics_and_unavailable_reasons(
    live: Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    live.sync(full=True)
    at = _app(live, monkeypatch)
    assert not at.exception
    assert not any("reconciliation status" in s.value for s in at.success)  # nothing before click
    _calculate(at)
    assert not at.exception and not at.error
    assert any("reconciliation status **Reconciled**" in s.value for s in at.success)
    labels = {m.label: m.value for m in at.metric}
    assert labels["Net invested capital"].startswith("amount 650.00")
    assert "per year" in labels["Money-weighted return (XIRR)"]
    unavailable = " ".join(i.value for i in at.info)
    assert "Total time-weighted return: unavailable" in unavailable
    assert "Portfolio equity curve: unavailable" in unavailable
    assert "Positions" in labels and "Cash ledger" in labels


def test_dashboard_shows_blockers_not_numbers_when_reconciliation_fails(
    env: Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("api.routes.performance._live_snapshot", lambda: (snapshot(qty=3.0), None))
    env.sync(full=True)
    at = _app(env, monkeypatch)
    _calculate(at)
    assert not at.exception
    labels = {m.label for m in at.metric}
    assert "Money-weighted return (XIRR)" not in labels
    assert "Net gain on invested capital" not in labels
    text = " ".join(w.value for w in at.warning)
    assert "Money-weighted return (XIRR): not verified" not in text  # unavailable, not partial
    assert any("do not reconcile" in i.value for i in at.info)


def test_dashboard_wrong_token_is_a_clear_error(live: Env, monkeypatch: pytest.MonkeyPatch) -> None:
    at = _app(live, monkeypatch)
    _calculate(at, token="wrong-token")
    assert any(
        "Could not calculate" in e.value and "Missing or invalid" in e.value for e in at.error
    )
    assert not any(m.label == "Net invested capital" for m in at.metric)
