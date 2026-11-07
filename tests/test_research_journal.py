"""Manual research journal tests: authentication, persistence, retrospective flag and outcomes."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

from sqlalchemy import select

from marketops.persistence.models import research_journal
from tests.history_fixtures import TOKEN, Env


def _idea() -> dict[str, object]:
    return {
        "symbol": "aapl",
        "decision_day": datetime.now(UTC).date().isoformat(),
        "thesis": "Illustrative historical breakout after consolidation",
        "risk_note": "Volume may reverse and stop execution is not guaranteed",
        "hypothetical_entry_gbp": "100.00",
        "hypothetical_units": "2.5",
        "planned_stop_gbp": "92.00",
    }


def _headers() -> dict[str, str]:
    return {"X-MarketOps-Token": TOKEN}


def test_journal_is_private_even_for_synthetic_ideas(env: Env) -> None:
    c = env.client
    assert c.get("/journal/ideas").status_code == 401
    assert c.post("/journal/ideas", json=_idea()).status_code == 401
    denied = c.post("/journal/ideas", json=_idea(), headers={"X-MarketOps-Token": "wrong"})
    assert denied.status_code == 401
    assert c.get("/journal/ideas", headers=_headers()).json() == []


def test_create_list_close_and_no_double_outcome(env: Env) -> None:
    c = env.client
    response = c.post("/journal/ideas", json=_idea(), headers=_headers())
    assert response.status_code == 201, response.text
    idea = response.json()
    assert idea["status"] == "open"
    assert idea["symbol"] == "AAPL"
    assert idea["retrospective"] is False
    assert idea["hypothetical_gross_pnl_gbp"] is None
    assert "not an actual order" in idea["limitation"]
    rows = c.get("/journal/ideas", headers=_headers()).json()
    assert len(rows) == 1 and rows[0]["id"] == idea["id"]
    payload = {
        "outcome_day": datetime.now(UTC).date().isoformat(),
        "hypothetical_exit_gbp": "110.00",
        "outcome_note": "Manual research outcome, not a real fill",
    }
    result = c.post(f"/journal/ideas/{idea['id']}/close", json=payload, headers=_headers())
    assert result.status_code == 200, result.text
    final = result.json()
    assert final["status"] == "closed"
    assert Decimal(final["hypothetical_gross_pnl_gbp"]) == Decimal("25.000")
    again = c.post(f"/journal/ideas/{idea['id']}/close", json=payload, headers=_headers())
    assert again.status_code == 409
    with env.engine.connect() as con:
        assert con.execute(select(research_journal.c.status)).scalar_one() == "closed"


def test_retrospective_is_explicit(env: Env) -> None:
    data = _idea()
    data["decision_day"] = (datetime.now(UTC).date() - timedelta(days=30)).isoformat()
    result = env.client.post("/journal/ideas", json=data, headers=_headers())
    assert result.status_code == 201
    assert result.json()["retrospective"] is True


def test_invalid_fields_rejected(env: Env) -> None:
    cases = [
        {"planned_stop_gbp": "120.0"},
        {"hypothetical_entry_gbp": "-1"},
        {"hypothetical_units": "0"},
        {"decision_day": (datetime.now(UTC).date() + timedelta(days=3)).isoformat()},
        {"symbol": "../secret"},
        {"thesis": "x"},
    ]
    for updates in cases:
        res = env.client.post("/journal/ideas", json=_idea() | updates, headers=_headers())
        assert res.status_code == 422, (updates, res.text)
    assert env.client.get("/journal/ideas", headers=_headers()).json() == []


def test_outcome_must_be_after_decision_and_not_in_future(env: Env) -> None:
    idea = _idea()
    idea["decision_day"] = datetime.now(UTC).date().isoformat()
    res = env.client.post("/journal/ideas", json=idea, headers=_headers())
    key = res.json()["id"]
    past = (datetime.now(UTC).date() - timedelta(days=1)).isoformat()
    future = (datetime.now(UTC).date() + timedelta(days=1)).isoformat()
    for day in (past, future):
        result = env.client.post(
            f"/journal/ideas/{key}/close",
            json={
                "outcome_day": day,
                "hypothetical_exit_gbp": "101",
                "outcome_note": "Illustrative outcome text",
            },
            headers=_headers(),
        )
        assert result.status_code == 422
    missing = env.client.post(
        "/journal/ideas/999999/close",
        json={
            "outcome_day": datetime.now(UTC).date().isoformat(),
            "hypothetical_exit_gbp": "101",
            "outcome_note": "Illustrative outcome text",
        },
        headers=_headers(),
    )
    assert missing.status_code == 404
