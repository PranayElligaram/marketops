"""Synthetic demo, benchmark in the valuation report, API demo route, view helpers, dashboard."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from api.main import app
from marketops.demo import build_demo
from marketops.valuation.report import HistoricalValuation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
import perf_view as pv  # noqa: E402


@pytest.fixture(scope="module")
def demo() -> HistoricalValuation:
    return build_demo()


def metric(v: HistoricalValuation, name: str) -> Any:
    assert v.benchmark is not None
    return next(m for m in v.benchmark.metrics if m.name == name)


def test_demo_is_deterministic_reconciled_and_labelled_synthetic(demo: HistoricalValuation) -> None:
    again = build_demo()
    assert demo.model_dump() == again.model_dump()
    assert demo.overall_status == "verified" and demo.reconciliation.status == "match"
    assert demo.notes[0].startswith("SYNTHETIC")
    assert demo.benchmark is not None and demo.benchmark.source is not None
    assert "synthetic" in demo.benchmark.source.symbol and demo.benchmark.source.sha256 is None


def test_demo_benchmark_is_available_and_like_for_like(demo: HistoricalValuation) -> None:
    b = demo.benchmark
    assert b is not None and b.status == "available" and not b.reasons
    # US-holiday first deposit (2023-01-02) rolls to the next priced day, so the window starts then
    assert b.window_start == "2023-01-03"
    nav = {p.date: p.nav for p in demo.series}
    for p in b.series:
        assert p.portfolio_value == pytest.approx(nav[p.date])
    assert b.series[-1].net_invested == metric(demo, "net_invested_capital").value == 13200
    # the benchmark's own value path obeys value_t = units * price: no negative value anywhere
    assert all(p.benchmark_value >= 0 for p in b.series)
    twr = metric(demo, "twr_difference").value
    assert twr == pytest.approx(
        metric(demo, "portfolio_twr").value - metric(demo, "benchmark_twr").value
    )
    assert metric(demo, "value_difference").value == pytest.approx(
        metric(demo, "portfolio_value").value - metric(demo, "benchmark_value").value
    )
    # the benchmark TWR equals the chained benchmark-price relative on the compared days
    assert b.series[-1].benchmark_twr == metric(demo, "benchmark_twr").value


def test_report_without_benchmark_data_marks_comparison_unavailable_with_reason(
    tmp_path: Path,
) -> None:
    from marketops.valuation.benchmark_data import load_benchmark

    bi = load_benchmark(tmp_path, None, "GBP")
    assert bi.problems and not bi.prices


def test_public_demo_route_needs_no_token_and_returns_benchmark() -> None:
    r = TestClient(app).get("/performance/demo")
    assert r.status_code == 200
    body = r.json()
    assert body["benchmark"]["status"] == "available" and body["notes"][0].startswith("SYNTHETIC")
    # private routes stay protected
    assert TestClient(app).get("/performance/valuation").status_code in (401, 403)


# ------------------------------------------------------------------ presentation helpers
def test_comparison_kpis_format_values_from_the_api_block(demo: HistoricalValuation) -> None:
    bench = demo.benchmark.model_dump() if demo.benchmark else {}
    rows = pv.build_comparison_kpis(bench)
    flat = {k.key: k for r in rows for k in r}
    assert len(rows) == 3 and all(len(r) == 4 for r in rows)
    assert flat["twr_difference"].value.endswith(" pp")
    assert flat["value_difference"].value.startswith(("+£", "-£"))
    assert flat["portfolio_max_drawdown"].value.startswith("-")
    assert all(k.tone == "ok" for k in flat.values())


def test_unavailable_metric_becomes_unavailable_kpi_with_reason() -> None:
    bench = {
        "currency": "GBP",
        "metrics": [
            {
                "name": "portfolio_annualised_twr",
                "value": None,
                "status": "unavailable",
                "blockers": ["Period shorter than one year."],
            }
        ],
    }
    k = next(x for r in pv.build_comparison_kpis(bench) for x in r if "annualised_twr" in x.key)
    assert k.value == "—" and k.tone == "unavailable" and "shorter" in k.note


def test_window_rebases_returns_exactly_and_keeps_full_history_drawdown(
    demo: HistoricalValuation,
) -> None:
    df = pv.comparison_frame(demo.benchmark.model_dump() if demo.benchmark else {})
    win = pv.window_comparison(df, "3M")
    assert len(win) < len(df) and win["portfolio_twr"].iloc[0] == 0.0
    first = df[df["date"] == win["date"].iloc[0]].iloc[0]
    last = df.iloc[-1]
    assert win["benchmark_twr"].iloc[-1] == pytest.approx(
        (1 + last["benchmark_twr"]) / (1 + first["benchmark_twr"]) - 1
    )
    assert list(win["portfolio_drawdown"]) == list(
        df[df["date"] >= win["date"].iloc[0]]["portfolio_drawdown"]
    )
    assert len(pv.window_comparison(df, "ALL")) == len(df)
    for p in pv.PERIODS:  # every selectable window renders a chart spec
        w = pv.window_comparison(df, p)
        assert pv.comparison_value_chart(w, "GBP").to_dict()
        assert pv.comparison_twr_chart(w).to_dict() and pv.comparison_drawdown_chart(w).to_dict()


# ------------------------------------------------------------------ dashboard (AppTest)
DASHBOARD = str(Path(__file__).resolve().parents[1] / "app" / "dashboard.py")


def _app(monkeypatch: pytest.MonkeyPatch) -> Any:
    from streamlit.testing.v1 import AppTest

    client = TestClient(app)

    def fake_get(url: str, **kw: Any) -> Any:
        if "/health" in url:
            return type("R", (), {"status_code": 200, "json": lambda self: {"status": "ready"}})()
        return client.get(
            url.removeprefix("http://localhost:8001"),
            params=kw.get("params"),
            headers=kw.get("headers"),
        )

    monkeypatch.setattr(httpx, "get", fake_get)
    return AppTest.from_file(DASHBOARD, default_timeout=60).run()


def test_dashboard_synthetic_demo_renders_benchmark_without_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    at = _app(monkeypatch)
    assert not at.exception
    next(b for b in at.button if b.label == "Load synthetic demo").click()
    at.run()
    assert not at.exception and not at.error
    assert any("Synthetic demonstration" in w.value for w in at.warning)
    assert "Portfolio vs S&P 500 benchmark" in {s.value for s in at.subheader}
    labels = {m.label for m in at.metric}
    assert {"Portfolio vs S&P 500", "Value vs S&P 500", "S&P 500 max drawdown"} <= labels
    assert any("How to read this comparison" in e.label for e in at.expander)


def test_dashboard_shows_exact_reason_when_comparison_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import api.routes.demo as demo_route

    broken = build_demo()
    assert broken.benchmark is not None
    broken.benchmark.status = "unavailable"
    broken.benchmark.reasons = ["Benchmark prices end on 2024-01-05 but the portfolio is later."]
    monkeypatch.setattr(demo_route, "_cache", broken)
    at = _app(monkeypatch)
    next(b for b in at.button if b.label == "Load synthetic demo").click()
    at.run()
    assert not at.exception
    assert any(
        "Comparison unavailable" in w.value and "end on 2024-01-05" in w.value for w in at.warning
    )
