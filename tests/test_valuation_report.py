"""Valuation report assembly, reconciliation, API authentication and dashboard rendering."""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
import pytest

from marketops.valuation.cache import store_series
from marketops.valuation.report import HistoricalValuation, build_valuation
from tests.history_fixtures import HDR, KEY, SECRET, TOKEN, Env
from tests.test_perf_report import snapshot
from tests.valuation_fixtures import bars, cash_tx, dividend, fill, fx_table, series, utc

START = date(2024, 1, 1)  # Monday
AS_OF = datetime(2024, 6, 28, 18, tzinfo=UTC)  # Friday


def _closes(n: int) -> list[float]:
    """A path with one 10% fall so drawdown is non-trivial: 10 up to day 50, then 9 ..."""
    return [10.0 + 0.01 * i if i < 50 else 9.0 + 0.01 * i for i in range(n)]


def _run(
    *,
    tx: list[Any] | None = None,
    fills: list[Any] | None = None,
    divs: list[Any] | None = None,
    frame: pd.DataFrame | None = None,
    verified: bool = True,
    qty: float = 10.0,
    total: float | None = None,
    fx: Any = None,
    as_of: datetime = AS_OF,
    ledger_cash: float | None = None,
) -> HistoricalValuation:
    frame = frame if frame is not None else bars(START, _closes(130))
    tx = tx if tx is not None else [cash_tx(1, utc(START), "deposit", "1000")]
    fills = (
        fills if fills is not None else [fill(1, utc(date(2024, 1, 2)), "buy", "10", "80", 10.0)]
    )
    last_close = float(frame["close"].iloc[-1])
    return build_valuation(
        fills=fills,
        tx=tx,
        divs=divs or [],
        ccy="GBP",
        as_of=as_of,
        price_root=None,
        fx=fx or fx_table(days=400),
        overrides=None,
        history_verified=verified,
        ledger_cash=ledger_cash if ledger_cash is not None else 920.0,
        broker_positions={"AAA_US_EQ": qty},
        broker_values={"AAA_US_EQ": qty * last_close * 0.8},
        broker_total=total if total is not None else 920.0 + qty * last_close * 0.8,
        series_override={"AAA": series("AAA", frame)},
    )


def metric(v: HistoricalValuation, name: str) -> Any:
    return next(m for m in v.metrics if m.name == name)


def test_verified_account_has_exact_nav_twr_risk_and_periods() -> None:
    v = _run()
    assert v.overall_status == "verified"
    nav = metric(v, "portfolio_nav")
    close = _closes(130)[-1]
    # 10 shares x last close x 0.8 (USD->GBP via the synthetic table) + 920 cash
    assert nav.status == "available" and nav.value == pytest.approx(920 + 10 * close * 0.8)
    assert metric(v, "net_invested_capital").value == 1000
    assert metric(v, "net_gain_on_invested_capital").value == pytest.approx(nav.value - 1000)
    # TWR equals the chain of daily NAV ratios (no external flows after the first day)
    navs = [p.nav for p in v.series if p.nav is not None][1:]
    first = [p.nav for p in v.series if p.nav is not None][0]
    expected = navs[-1] / first - 1
    assert metric(v, "time_weighted_return").value == pytest.approx(expected)
    assert metric(v, "annualised_twr").status == "unavailable"  # shorter than a year
    assert metric(v, "annualised_volatility").status == "available"
    assert metric(v, "max_drawdown").value <= 0
    assert v.reconciliation.status == "match" and v.reconciliation.quantity_status == "reconciled"
    assert v.coverage.unvalued_days == 0


def test_a_large_deposit_is_not_a_gain_and_earns_nothing_on_its_day() -> None:
    dep_day = date(2024, 3, 4)
    more = _run(
        tx=[cash_tx(1, utc(START), "deposit", "1000"), cash_tx(2, utc(dep_day), "deposit", "5000")]
    )
    assert metric(more, "net_invested_capital").value == 6000
    by = {p.date: p for p in more.series}
    prev, day = by["2024-03-01"], by[dep_day.isoformat()]
    assert prev.nav is not None and day.nav is not None
    assert prev.twr_index is not None and day.twr_index is not None
    # NAV jumps by the deposit plus the day's market move on the old capital only
    assert day.twr_index / prev.twr_index == pytest.approx((day.nav - 5000) / prev.nav)
    naive = (more.series[-1].nav or 0) / (more.series[0].nav or 1) - 1
    assert naive > 4 and metric(more, "time_weighted_return").value < 0.05  # 5000 is not profit
    assert more.overall_status == "partial"  # ledger cash (920) no longer matches this history


def test_withdrawal_is_not_a_loss() -> None:
    v = _run(
        tx=[
            cash_tx(1, utc(START), "deposit", "1000"),
            cash_tx(2, utc(date(2024, 3, 4)), "withdrawal", "-300"),
        ],
        ledger_cash=620.0,
        total=620.0 + 10 * _closes(130)[-1] * 0.8,
    )
    assert metric(v, "net_invested_capital").value == 700
    base = metric(_run(), "time_weighted_return").value
    assert metric(v, "time_weighted_return").value == pytest.approx(base, abs=0.005)


def test_unreconciled_history_never_reports_a_verified_return() -> None:
    v = _run(verified=False)
    assert v.overall_status == "partial"
    twr_metric = metric(v, "time_weighted_return")
    assert twr_metric.status == "partial" and twr_metric.value is not None and twr_metric.blockers


def test_share_count_mismatch_is_a_reconciliation_failure() -> None:
    v = _run(qty=11.0)
    assert v.reconciliation.quantity_status == "mismatch" and v.overall_status == "partial"
    assert any("quantity differs" in r for r in v.reconciliation.reasons)


def test_large_nav_gap_to_the_live_total_is_a_mismatch() -> None:
    v = _run(total=5000.0)
    assert v.reconciliation.status == "mismatch" and v.overall_status == "partial"
    live_total = (920 + 10 * _closes(130)[-1] * 0.8) * 1.03
    assert _run(total=live_total).reconciliation.status == "drift"


def test_cash_disagreement_with_the_ledger_is_reported() -> None:
    v = _run(ledger_cash=300.0)
    assert any("cash differs" in r for r in v.reconciliation.reasons)


def test_missing_fx_for_part_of_the_period_leaves_gaps_not_numbers() -> None:
    fx = fx_table(days=45)  # rates stop in early February
    v = _run(fx=fx)
    assert v.coverage.unvalued_days > 50
    assert "missing_fx" in v.coverage.unvalued_reasons
    assert v.overall_status == "partial"
    assert all(p.nav is None for p in v.series[-5:])
    assert v.coverage.fx_currencies


def test_missing_prices_from_a_date_leave_a_gap_and_split_the_segments() -> None:
    frame = bars(START, _closes(130))
    frame = frame[(frame["date"] < "2024-03-04") | (frame["date"] > "2024-03-22")].reset_index(
        drop=True
    )
    v = _run(frame=frame)
    assert v.coverage.unvalued_days >= 10
    t = metric(v, "time_weighted_return")
    assert t.status == "partial" and any("segments" in b for b in t.blockers)
    # the TWR index restarts after the gap instead of bridging it
    pts = [p.twr_index for p in v.series if p.twr_index is not None]
    assert pts and min(pts) > 0


def test_fully_sold_instrument_is_mapped_and_valued_historically() -> None:
    fills = [
        fill(1, utc(date(2024, 1, 2)), "buy", "10", "80", 10.0),
        fill(2, utc(date(2024, 2, 1)), "sell", "10", "81.6", 10.2),
    ]
    v = _run(fills=fills, qty=0.0, ledger_cash=1001.6, total=1001.6)
    row = v.mapping[0]
    assert row.status == "verified" and row.fills_checked == 2
    assert v.coverage.unvalued_days == 0
    assert metric(v, "portfolio_nav").value == pytest.approx(1001.6)


def test_unverifiable_symbol_blocks_the_instrument() -> None:
    frame = bars(START, [50.0] * 130)  # wrong instrument: far from the fill price
    v = _run(frame=frame)
    assert v.mapping[0].status == "unverified" and v.overall_status == "partial"
    assert v.coverage.valued_days < v.coverage.weekdays


def test_dividends_raise_nav_by_exactly_the_cash_received() -> None:
    d = dividend(1, utc(date(2024, 5, 1)), "7.5")
    v = _run(divs=[d], ledger_cash=927.5, total=927.5 + 10 * _closes(130)[-1] * 0.8)
    plain = _run()
    assert metric(v, "portfolio_nav").value == pytest.approx(
        metric(plain, "portfolio_nav").value + 7.5
    )
    assert v.overall_status == "verified"


def test_no_history_or_currency_is_unavailable() -> None:
    v = _run(tx=[], fills=[])
    assert v.overall_status == "unavailable" and v.series == []


def test_period_tables_flag_incomplete_periods() -> None:
    v = _run()
    assert v.monthly and not v.monthly[0].complete  # first month starts mid-way
    assert any(m.complete for m in v.monthly)
    assert v.annual and v.annual[0].complete is False


# ------------------------------------------------------------------ API
def _install_market_cache(tmp: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Public-style synthetic bars covering the standard fixture history up to today."""
    root = tmp / "market"
    today = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    days = pd.bdate_range("2024-02-01", today)
    close = [10.0 if d < pd.Timestamp("2024-04-01") else 65.0 for d in days]
    frame = pd.DataFrame(
        {
            "date": days,
            "open": close,
            "high": [c * 1.01 for c in close],
            "low": [c * 0.99 for c in close],
            "close": close,
            "volume": 1.0,
            "dividend": 0.0,
            "split": 1.0,
        }
    )
    store_series(root / "prices", "AAA", frame, "USD")
    rows = ["Date,USD,GBP,"] + [
        f"{d.date()},1.25,1.0," for d in pd.bdate_range("2024-01-01", today)
    ]
    (root / "ecb_eurofxref_hist.csv").write_text(
        "\n".join(rows[:1] + rows[:0:-1]), encoding="utf-8"
    )
    monkeypatch.setenv("MARKETOPS_MARKET_CACHE_DIR", str(root))
    from marketops.config import get_settings

    get_settings.cache_clear()
    return root


@pytest.fixture
def live(env: Env, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Env:
    monkeypatch.setattr("api.routes.performance._live_snapshot", lambda: (snapshot(), None))
    _install_market_cache(tmp_path, monkeypatch)
    return env


def test_valuation_routes_need_the_token(live: Env) -> None:
    assert live.client.get("/performance/valuation").status_code == 401
    wrong = {"X-MarketOps-Token": "wrong-token-value-123"}
    assert live.client.get("/performance/valuation", headers=wrong).status_code == 401
    assert live.client.post("/performance/valuation/refresh").status_code == 401


def test_valuation_denied_when_no_token_configured(
    live: Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    from marketops.config import get_settings

    monkeypatch.delenv("MARKETOPS_AGENT_API_TOKEN")
    get_settings.cache_clear()
    assert live.client.get("/performance/valuation", headers=HDR).status_code == 403


def test_valuation_without_history_is_404(live: Env) -> None:
    assert live.client.get("/performance/valuation", headers=HDR).status_code == 404


def test_valuation_route_returns_a_verified_reconciled_valuation(live: Env) -> None:
    live.sync(full=True)
    r = live.client.get("/performance/valuation", headers=HDR)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["account_currency"] == "GBP"
    assert body["overall_status"] == "verified", body["reconciliation"]
    assert body["reconciliation"]["status"] == "match"
    assert body["mapping"][0]["status"] == "verified"
    assert body["series"][0]["net_invested"] == 200  # first deposit
    names = {m["name"] for m in body["metrics"]}
    assert {"portfolio_nav", "time_weighted_return", "net_invested_capital"} <= names
    for secret in (KEY, SECRET, TOKEN):
        assert secret not in r.text


def test_valuation_without_cached_prices_is_partial_not_invented(
    env: Env, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("api.routes.performance._live_snapshot", lambda: (snapshot(), None))
    monkeypatch.setenv("MARKETOPS_MARKET_CACHE_DIR", str(tmp_path / "empty"))
    from marketops.config import get_settings

    get_settings.cache_clear()
    env.sync(full=True)
    body = env.client.get("/performance/valuation", headers=HDR).json()
    assert body["overall_status"] != "verified"
    assert body["mapping"][0]["status"] == "unverified"
    twr_metric = next(m for m in body["metrics"] if m["name"] == "time_weighted_return")
    assert twr_metric["status"] != "available"


def test_refresh_downloads_only_symbols_and_fx(
    live: Env, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    live.sync(full=True)
    seen: list[str] = []

    class FakeProvider:
        name = "fake"

        def fetch_daily(self, symbol: str, start: date, end: date) -> pd.DataFrame:
            seen.append(symbol)
            f = bars(start, [10.0] * 5)
            f.attrs["currency"] = "USD"
            return f

    class FakeClient:
        def get(self, url: str, params: Any = None) -> Any:
            day = str(pd.Timestamp.now(tz="UTC").date())
            body = {"base": "EUR", "rates": {day: {"USD": 1.25, "GBP": 1.0}}}
            return httpx.Response(200, json=body)

    monkeypatch.setattr("api.routes.performance.YahooProvider", FakeProvider)
    monkeypatch.setattr("marketops.valuation.fx.httpx.Client", lambda **kw: FakeClient())
    r = live.client.post("/performance/valuation/refresh", headers=HDR)
    assert r.status_code == 200
    assert seen == ["AAA"] and r.json()["AAA"].startswith("ok")
    assert r.json()["FX"].startswith("ok")


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


def test_dashboard_renders_valuation_metrics_and_charts(
    live: Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    live.sync(full=True)
    at = _app(live, monkeypatch)
    assert not at.exception
    at.text_input(key="tok_perf").set_value(TOKEN).run()
    next(b for b in at.button if b.label == "Calculate historical valuation").click()
    at.run()
    assert not at.exception and not at.error
    assert any("reconciliation status **Reconciled**" in s.value for s in at.success)
    labels = {m.label for m in at.metric}
    assert {"Portfolio value", "Net investment gain", "Total time-weighted return"} <= labels


def test_dashboard_shows_reasons_when_prices_are_missing(
    env: Env, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("api.routes.performance._live_snapshot", lambda: (snapshot(), None))
    monkeypatch.setenv("MARKETOPS_MARKET_CACHE_DIR", str(tmp_path / "none"))
    from marketops.config import get_settings

    get_settings.cache_clear()
    env.sync(full=True)
    at = _app(env, monkeypatch)
    at.text_input(key="tok_perf").set_value(TOKEN).run()
    next(b for b in at.button if b.label == "Calculate historical valuation").click()
    at.run()
    assert not at.exception
    assert not any("reconciliation status **Reconciled**" in s.value for s in at.success)
    assert any(
        "unverified" in w.value.lower() or "could not" in w.value.lower() for w in at.warning
    )
