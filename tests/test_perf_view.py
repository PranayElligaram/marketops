"""Presentation layer of the Historical Performance tab, driven by synthetic valuations only."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from tests.test_valuation_report import _run

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
import perf_view as pv  # noqa: E402

DASHBOARD = str(Path(__file__).resolve().parents[1] / "app" / "dashboard.py")


def _val(**kw: Any) -> dict[str, Any]:
    return _run(**kw).model_dump(mode="json")


def _perf(rate: float | None = 0.05) -> dict[str, Any]:
    metric: dict[str, Any] = {
        "name": "money_weighted_return_xirr",
        "status": "available" if rate is not None else "unavailable",
        "value": {"annualised_rate": rate} if rate is not None else None,
        "blockers": [] if rate is not None else ["Live snapshot did not reconcile."],
    }
    return {"account_currency": "GBP", "metrics": [metric]}


def _flat(rows: list[list[pv.Kpi]]) -> dict[str, pv.Kpi]:
    return {k.key: k for r in rows for k in r}


def _metric(val: dict[str, Any], name: str) -> float:
    return float(next(m["value"] for m in val["metrics"] if m["name"] == name))


def test_kpis_are_two_rows_of_four_and_match_the_api_values() -> None:
    val = _val()
    rows = pv.build_kpis(val, _perf())
    assert [len(r) for r in rows] == [4, 4]
    k = _flat(rows)
    assert k["portfolio_nav"].value == pv.money(_metric(val, "portfolio_nav"), "GBP")
    assert k["net_invested_capital"].value == "£1,000.00"
    assert k["net_gain_on_invested_capital"].value.startswith(("+£", "-£"))
    assert k["time_weighted_return"].value == f"{_metric(val, 'time_weighted_return'):+.2%}"
    assert k["max_drawdown"].value == f"{_metric(val, 'max_drawdown'):.2%}"
    assert k["money_weighted_return_xirr"].value == "+5.00% / yr"
    assert all(x.tone == "ok" for key, x in k.items() if key != "annualised_twr")
    assert k["annualised_twr"].tone == "unavailable" and k["annualised_twr"].value == "—"
    assert "Live snapshot" in k["money_weighted_return_xirr"].source
    assert "Historical closes" in k["portfolio_nav"].source


def test_no_raw_internal_names_in_kpi_labels() -> None:
    labels = [k.label for r in pv.build_kpis(_val(), _perf()) for k in r]
    assert all("_" not in label for label in labels)
    assert pv.label_for("some_unknown_metric") == "Some unknown metric"


def test_xirr_pending_and_unavailable_states_do_not_invent_a_number() -> None:
    assert _flat(pv.build_kpis(_val(), None))["money_weighted_return_xirr"].tone == "pending"
    unavailable = _flat(pv.build_kpis(_val(), _perf(rate=None)))["money_weighted_return_xirr"]
    assert unavailable.value == "—" and "did not reconcile" in unavailable.note


def test_unreconciled_values_are_flagged_not_presented_as_reconciled() -> None:
    val = _val(verified=False)
    assert val["overall_status"] != "verified"
    tones = {k.tone for r in pv.build_kpis(val, None) for k in r if k.value != "—"}
    assert "unreconciled" in tones and "ok" not in tones
    assert pv.RECON_LABEL["verified"] == "Reconciled"
    assert "verified" not in " ".join(pv.RECON_LABEL.values()).lower()


def test_empty_valuation_has_dashes_and_an_empty_chart_frame() -> None:
    val = _val(tx=[], fills=[])
    k = _flat(pv.build_kpis(val, None))
    assert k["portfolio_nav"].value == "—" and k["time_weighted_return"].value == "—"
    empty = pv.series_frame({"series": []})
    assert empty.empty and pv.filter_period(empty, "1M").empty


def test_period_filter_only_changes_the_window_never_the_numbers() -> None:
    val = _val()
    full = pv.series_frame(val)
    before = pv.build_kpis(val, None)
    for period, days in (("1M", 32), ("3M", 93), ("6M", 186), ("1Y", 366)):
        shown = pv.filter_period(full, period)
        assert 0 < len(shown) <= len(full)
        assert (shown["date"].max() - shown["date"].min()).days <= days
        assert shown["date"].max() == full["date"].max()
        merged = shown.merge(full, on="date", suffixes=("", "_f"))
        pd.testing.assert_series_equal(merged["twr"], merged["twr_f"], check_names=False)
        pd.testing.assert_series_equal(merged["drawdown"], merged["drawdown_f"], check_names=False)
    assert len(pv.filter_period(full, "1M")) < len(full)
    assert pv.filter_period(full, "ALL").equals(full)
    assert pv.build_kpis(val, None) == before


def test_series_frame_matches_api_points_and_end_values_match_the_kpis() -> None:
    val = _val()
    f = pv.series_frame(val)
    assert f["nav"].iloc[0] == f["nav"].dropna().iloc[0]  # starts at the first valued day
    assert len(f) == len([p for p in val["series"] if p["nav"] is not None])
    assert f["nav"].dropna().iloc[-1] == pytest.approx(_metric(val, "portfolio_nav"))
    assert f["twr"].dropna().iloc[-1] == pytest.approx(_metric(val, "time_weighted_return"))
    assert f["drawdown"].min() == pytest.approx(_metric(val, "max_drawdown"))


def test_charts_have_gbp_axis_percent_axes_tooltips_and_sensible_heights() -> None:
    f = pv.series_frame(_val())
    value, twr, dd = pv.value_chart(f, "GBP"), pv.twr_chart(f), pv.drawdown_chart(f)
    text = str(value.to_dict())
    assert "'£'" in text and "Net invested capital" in text and "tooltip" in text
    assert value.to_dict()["height"] == 340
    assert twr.to_dict()["height"] == 260 and dd.to_dict()["height"] == 200
    assert "'format': '%'" in str(twr.to_dict()) and "'format': '%'" in str(dd.to_dict())
    assert "+.2%" in str(twr.to_dict())


def test_period_tables_are_sorted_formatted_and_mark_partial_periods() -> None:
    val = _val()
    t = pv.period_table(val["monthly"])
    assert list(t.columns) == ["Period", "Return", "Days", "Coverage"]
    assert list(t["Period"]) == sorted(t["Period"], reverse=True)
    assert "Partial period" in set(t["Coverage"]) and "Complete" in set(t["Coverage"])
    html = pv.style_period_table(t).to_html()
    assert f"{t['Return'].iloc[0]:+.2%}" in html


def test_money_and_percent_formatting_is_consistent() -> None:
    assert pv.money(-1234.5, "GBP") == "-£1,234.50"
    assert pv.money(5, "CHF") == "CHF 5.00"
    assert pv.pct(0.1234) == "12.34%" and pv.pct(-0.1, signed=True) == "-10.00%"


# ----------------------------------------------------------------- rendered dashboard
@pytest.fixture(autouse=True)
def _offline_api(monkeypatch: pytest.MonkeyPatch) -> None:
    """The dashboard gates on GET /health/ready; AppTest runs it in-process with no API.
    Answer only that probe and fail every other call as the real client would offline."""

    def fake_get(url: str, **_: Any) -> httpx.Response:
        if url.endswith("/health/ready"):
            return httpx.Response(200, json={"status": "ready"})
        raise httpx.ConnectError("offline test: no API", request=httpx.Request("GET", url))

    def fake_post(url: str, **_: Any) -> httpx.Response:
        raise httpx.ConnectError("offline test: no API", request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx, "get", fake_get)
    monkeypatch.setattr(httpx, "post", fake_post)


def _render(val: dict[str, Any] | None, perf: dict[str, Any] | None = None) -> AppTest:
    at = AppTest.from_file(DASHBOARD, default_timeout=60)
    if val is not None:
        at.session_state["val"] = val
    if perf is not None:
        at.session_state["perf"] = perf
    return at.run()


def _charts(at: AppTest) -> Any:
    return at.tabs[2].get("vega_lite_chart")


def test_rendered_dashboard_has_eight_kpis_charts_tables_and_expanders() -> None:
    val = _val()
    at = _render(val)
    assert not at.exception
    assert [m.label for m in at.metric][:8] == [
        k.label for r in pv.build_kpis(val, None) for k in r
    ]
    assert any("reconciliation status **Reconciled**" in s.value for s in at.success)
    assert len(_charts(at)) == 3
    assert len(at.dataframe) >= 2
    assert {
        "Reconciliation with the live account",
        "Data coverage",
        "Assumptions and conventions",
    } <= {e.label for e in at.expander}
    text = " ".join(x.value for x in [*at.markdown, *at.caption, *at.warning, *at.success])
    assert "verified" not in text.lower().replace("unverified", "")
    assert "portfolio_nav" not in text and "twr_flow_timing" not in text
    assert "different questions" in text


def test_rendered_partial_state_keeps_warnings_visible() -> None:
    at = _render(_val(verified=False))
    assert not at.exception
    assert any("Partially reconciled" in w.value for w in at.warning)


def test_rendered_unavailable_state_is_a_clear_message() -> None:
    at = _render(_val(tx=[], fills=[]))
    assert not at.exception
    assert any("Unavailable" in i.value for i in at.info)
    assert len(_charts(at)) == 0


def test_rendered_period_selection_changes_the_chart_but_not_the_kpis() -> None:
    at = _render(_val())
    before = [m.value for m in at.metric]
    n_all = _charts(at)[0].proto.spec
    at.get("button_group")[0].set_value("1M").run()
    assert not at.exception
    assert [m.value for m in at.metric] == before
    assert _charts(at)[0].proto.spec != n_all


def test_xirr_card_uses_the_performance_endpoint_value() -> None:
    at = _render(_val(), _perf(0.0731))
    assert any(m.label == "Money-weighted return (XIRR)" and "+7.31%" in m.value for m in at.metric)
