"""Security, reliability and bounded-execution checks for the research agent."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.routes import agent as agent_route
from marketops.agents import ollama
from marketops.agents.runner import run_research
from marketops.agents.tools import Toolbox
from marketops.config import Settings, get_settings
from tests.test_agent import Fake, calls, good_flow, settings, synthesis, tc  # noqa: F401

TOKEN = "t" * 24


# ------------------------------------------------------------------ tool surface
def test_tool_arguments_cannot_express_paths_urls_sql_or_orders(settings: Settings) -> None:  # noqa: F811
    allowed_fields = {"symbol", "days", "period"}
    for broker in ("none", "example"):
        for name, t in Toolbox(settings, broker).defs().items():  # type: ignore[arg-type]
            assert set(t.args.model_fields) <= allowed_fields, name
            assert t.args.model_config.get("extra") == "forbid", name
    names = " ".join(Toolbox(settings, "example").defs())  # type: ignore[arg-type]
    assert not any(w in names for w in ("order", "trade", "sql", "file", "http", "fetch", "exec"))


def test_garbage_tool_calls_are_contained(settings: Settings) -> None:  # noqa: F811
    bad = calls(None, {"function": None}, {"function": {"name": 5}}, {"function": {"arguments": 1}})  # type: ignore[arg-type]
    bad["tool_calls"].append({"function": {"name": "get_ml_signal", "arguments": "{" * 5000}})
    report, _ = run_research("AAA?", "none", settings, Fake(bad, synthesis()))
    assert report.tool_calls <= settings.agent_max_tool_calls
    assert all(t.status in ("rejected", "error", "limit", "ok") for t in report.tool_trace)


def test_hostile_tool_arguments_never_leave_the_allowlist(settings: Settings) -> None:  # noqa: F811
    hostile = calls(
        tc("get_price_summary", symbol="AAA", path="C:/Windows/win.ini"),
        tc("get_price_summary", symbol="AAA; DROP TABLE agent_run"),
        tc("place_order", symbol="AAA", side="buy"),
        tc("http_get", url="http://169.254.169.254/"),
        tc("run_sql", query="select * from agent_run"),
    )
    report, _ = run_research("AAA", "none", settings, Fake(hostile, synthesis()))
    oks = [t.tool for t in report.tool_trace if t.status == "ok" and t.origin == "llm"]
    assert oks == []
    assert sum(1 for t in report.tool_trace if t.status == "rejected") == 5


# ------------------------------------------------------------------ failure handling
@pytest.mark.parametrize(
    "script",
    [
        [ollama.OllamaUnavailable("down")],
        [ollama.ModelUnavailable("missing")],
        [ollama.LLMTimeout("slow")],
        [ollama.BadModelResponse("bad")],
        [calls(tc("get_ml_signal", symbol="AAA")), {"content": "not json"}],
        [calls(tc("get_ml_signal", symbol="AAA")), {"content": json.dumps({"verified_facts": 3})}],
        [calls(tc("get_ml_signal", symbol="AAA")), {"content": ""}],
    ],
)
def test_fallback_never_presents_itself_as_a_completed_llm_run(
    settings: Settings,  # noqa: F811
    script: list[Any],
) -> None:
    report, audit = run_research("Tell me about AAA", "none", settings, Fake(*script))
    assert report.mode == "deterministic_fallback"
    assert report.status == "completed_with_warnings" and audit["status"] == report.status
    assert report.llm_status != "ok"
    assert any("no model-written analysis" in w for w in report.warnings)
    assert report.interpretation == []
    assert {t.origin for t in report.tool_trace} >= {"fallback"}


def test_no_client_is_a_labelled_fallback(settings: Settings) -> None:  # noqa: F811
    report, _ = run_research("AAA?", "none", settings, None)
    assert report.mode == "deterministic_fallback" and report.llm_status == "not_configured"
    assert report.llm_model == "none"


def test_wall_clock_deadline_is_enforced(settings: Settings) -> None:  # noqa: F811
    s = settings.model_copy(update={"agent_timeout_s": 0.0})
    fake = good_flow()
    report, _ = run_research("AAA?", "none", s, fake)
    assert report.llm_status == "timeout" and report.mode == "deterministic_fallback"
    assert fake.calls == []  # no model call is started once the budget is spent


def test_ollama_client_timeouts_and_malformed_bodies() -> None:
    def client(handler: Any) -> ollama.OllamaClient:
        return ollama.OllamaClient(
            "http://127.0.0.1:11434", "m", transport=httpx.MockTransport(handler)
        )

    def slow(_: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow")

    with pytest.raises(ollama.LLMTimeout):
        client(slow).chat([])
    for body in (b"<html>", b"[]", b'{"message": "str"}', b'{"message": null}'):
        with pytest.raises(ollama.BadModelResponse):
            client(lambda r, b=body: httpx.Response(200, content=b)).chat([])
    with pytest.raises(ollama.BadModelResponse):
        client(lambda r: httpx.Response(500)).chat([])


def test_generation_is_bounded_per_call() -> None:
    seen: list[dict[str, Any]] = []

    def handler(r: httpx.Request) -> httpx.Response:
        seen.append(json.loads(r.content))
        return httpx.Response(200, json={"message": {"content": "{}"}})

    c = ollama.OllamaClient("http://127.0.0.1:11434", "m", transport=httpx.MockTransport(handler))
    c.chat([], tools=[{"type": "function"}])
    c.chat([], fmt={"type": "object"})
    assert seen[0]["options"]["num_predict"] <= 400 and seen[1]["options"]["num_predict"] <= 1200


# ------------------------------------------------------------------ protocol / cost control
def test_one_tool_round_then_one_synthesis_call(settings: Settings) -> None:  # noqa: F811
    fake = good_flow()
    report, _ = run_research("AAA?", "none", settings, fake)
    assert len(fake.calls) == 2 and report.timing.llm_calls == 2
    names = [t["function"]["name"] for t in fake.calls[0]["tools"]]
    assert "get_model_evaluation" not in names  # already retrieved as E1
    assert "E1" in fake.calls[0]["messages"][1]["content"]
    assert report.timing.tools_ms >= 0 and report.timing.verify_ms >= 0


def test_failed_tool_call_allows_another_round(settings: Settings) -> None:  # noqa: F811
    fake = Fake(
        calls(tc("get_price_summary", symbol="ZZZ")),
        calls(tc("get_price_summary", symbol="AAA")),
        synthesis(verified_facts=[{"text": "AAA has price data.", "evidence": ["E2"]}]),
    )
    report, _ = run_research("AAA?", "none", settings, fake)
    assert report.mode == "llm" and report.iterations == 2 and report.verified_facts


def test_tool_selection_turn_sees_compact_results_only(settings: Settings) -> None:  # noqa: F811
    fake = Fake(
        calls(tc("get_backtest", period="test"), tc("get_shap_importance")),
        synthesis(),
    )
    run_research("backtest?", "none", settings, fake)
    tool_msgs = [m for m in fake.calls[0]["messages"] if m["role"] == "tool"]
    big = json.loads(tool_msgs[0]["content"])
    assert big["retrieved"] is True and "untrusted_data" not in big  # payload withheld this turn
    assert big["evidence_id"] == "E2"
    synth_prompt = fake.calls[-1]["messages"][-1]["content"]
    assert '"cagr":0.1439' in synth_prompt  # full payload reaches the model once, compactly


def test_question_naming_a_symbol_without_research_falls_back(settings: Settings) -> None:  # noqa: F811
    report, _ = run_research("Tell me about AAA", "none", settings, Fake({"content": "no tools"}))
    assert report.llm_status == "model_retrieved_no_evidence"


def test_general_question_answers_from_preloaded_evaluation(settings: Settings) -> None:  # noqa: F811
    # REGRESSION (live 2026-10-28): the model asked only for the evaluation that was already
    # fetched as E1; that was treated as "no evidence" and a template replaced the answer.
    fake = Fake(
        calls(tc("get_model_evaluation")),
        synthesis(
            verified_facts=[
                {"text": "The held-out ROC-AUC of the model was 0.4961.", "evidence": ["E1"]}
            ]
        ),
    )
    report, _ = run_research("Does the model have an edge?", "none", settings, fake)
    assert report.mode == "llm" and report.verified_facts


def test_over_limit_claims_are_listed_not_dropped(settings: Settings) -> None:  # noqa: F811
    many = [{"text": f"AAA has price data {i}.", "evidence": ["E2"]} for i in range(6)]
    fake = Fake(calls(tc("get_price_summary", symbol="AAA")), synthesis(verified_facts=many))
    report, _ = run_research("AAA?", "none", settings, fake)
    assert len(report.verified_facts) == 4
    over = [r for r in report.rejected_claims if "limit" in r.reason]
    assert len(over) == 2 and report.status == "completed_with_warnings"


def test_synthesis_schema_bounds_output_length() -> None:
    from marketops.agents.runner import _Synthesis

    schema = _Synthesis.model_json_schema()
    assert schema["properties"]["verified_facts"]["maxItems"] <= 4
    assert schema["$defs"]["_Claims"]["properties"]["text"]["maxLength"] <= 240


# ------------------------------------------------------------------ access control and audit
@pytest.fixture
def broker_api(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> TestClient:  # noqa: F811
    class FakeBroker:
        def __init__(self, **_: Any) -> None: ...

        def get_positions(self) -> list[dict[str, Any]]:
            from marketops.portfolio.analytics import EXAMPLE_POSITIONS

            return EXAMPLE_POSITIONS

    monkeypatch.setattr("marketops.agents.tools.Trading212ReadOnlyClient", FakeBroker)
    monkeypatch.setenv("MARKETOPS_AGENT_ALLOW_BROKER_PORTFOLIO", "true")
    monkeypatch.setenv("MARKETOPS_T212_API_KEY", "k")
    monkeypatch.setenv("MARKETOPS_T212_API_SECRET", "s")
    get_settings.cache_clear()
    monkeypatch.setattr(
        agent_route,
        "get_client",
        lambda: Fake(
            calls(tc("get_portfolio_exposure")),
            synthesis(verified_facts=[{"text": "Largest weight is 60.0%.", "evidence": ["E2"]}]),
        ),
    )
    yield TestClient(app)
    get_settings.cache_clear()


def _post(c: TestClient, token: str | None) -> httpx.Response:
    h = {"X-MarketOps-Token": token} if token else {}
    return c.post(
        "/agent/research",
        json={"question": "concentration?", "portfolio_source": "trading212"},
        headers=h,
    )


def test_broker_research_refused_without_configured_token(broker_api: TestClient) -> None:
    r = _post(broker_api, "anything")
    assert r.status_code == 403 and "MARKETOPS_AGENT_API_TOKEN" in r.json()["detail"]


def test_broker_research_needs_matching_token(
    broker_api: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MARKETOPS_AGENT_API_TOKEN", TOKEN)
    get_settings.cache_clear()
    assert _post(broker_api, None).status_code == 401
    assert _post(broker_api, "wrong" * 5).status_code == 401
    ok = _post(broker_api, TOKEN)
    assert ok.status_code == 200
    body = ok.text
    assert "EXAMPLE_A" not in body and TOKEN not in body
    ev = next(e for e in ok.json()["evidence"] if e["tool"] == "get_portfolio_exposure")
    assert ev["data"] is None and ev["sha256"] == "withheld"


def test_audit_listing_is_token_gated_only_when_broker_enabled(
    broker_api: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MARKETOPS_AGENT_API_TOKEN", TOKEN)
    get_settings.cache_clear()
    assert broker_api.get("/agent/runs").status_code in (401, 403)
    monkeypatch.setenv("MARKETOPS_AGENT_ALLOW_BROKER_PORTFOLIO", "false")
    get_settings.cache_clear()
    assert broker_api.get("/agent/runs").status_code in (
        200,
        503,
    )  # no token needed, no broker data


def test_broker_audit_record_has_no_hashes_or_payloads(
    settings: Settings,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeBroker:
        def __init__(self, **_: Any) -> None: ...

        def get_positions(self) -> list[dict[str, Any]]:
            from marketops.portfolio.analytics import EXAMPLE_POSITIONS

            return EXAMPLE_POSITIONS

    monkeypatch.setattr("marketops.agents.tools.Trading212ReadOnlyClient", FakeBroker)
    s = settings.model_copy(
        update={"agent_allow_broker_portfolio": True, "t212_api_key": "k", "t212_api_secret": "s"}
    )
    fake = Fake(calls(tc("get_portfolio_exposure")), synthesis())
    report, audit = run_research("concentration?", "trading212", s, fake)
    ref = next(r for r in audit["evidence_refs"] if r["tool"] == "get_portfolio_exposure")
    assert ref["sha256"] == "withheld"
    assert next(e for e in report.evidence if e.sensitive).sha256 == "withheld"
    assert "EXAMPLE_A" not in json.dumps(audit)


def test_dev_oof_curve_overlapping_test_period_is_flagged(settings: Settings) -> None:  # noqa: F811
    # LIVE finding: the real artefact's dev_oof curve runs to 2026 although dev_end is 2021-12-31.
    path = next(__import__("pathlib").Path(settings.model_dir).glob("v1/backtest.json"))
    bt = json.loads(path.read_text())
    bt["equity_curves"]["dev_oof"]["model"]["2026-10-26"] = 3.0
    path.write_text(json.dumps(bt))
    tb = Toolbox(settings)
    flagged = tb.execute("get_backtest", {"period": "dev_oof"})[2]
    assert "overlaps the held-out test period" in flagged["period_warning"]
    assert tb.execute("get_backtest", {"period": "test"})[2]["period_warning"] is None
