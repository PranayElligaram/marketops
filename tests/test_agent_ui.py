"""Streamlit rendering test for the research agent tab (offline; HTTP is stubbed)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from marketops.agents.runner import run_research
from marketops.config import Settings
from tests.test_agent import good_flow, settings  # noqa: F401  (fixture reuse)

DASHBOARD = str(Path(__file__).resolve().parents[1] / "app" / "dashboard.py")


class _Resp:
    def __init__(self, data: Any, code: int = 200) -> None:
        self.data, self.status_code = data, code

    def json(self) -> Any:
        return self.data


def test_agent_tab_renders_report(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: F811
    from streamlit.testing.v1 import AppTest

    report, _ = run_research("AAA?", "example", settings, good_flow())
    body = json.loads(report.model_dump_json())

    def fake_get(url: str, **_: Any) -> _Resp:
        if url.endswith("/agent/status"):
            llm = {"ollama": "unavailable", "model": "m", "model_installed": False, "detail": "x"}
            return _Resp({"llm": llm})
        if "health" in url:
            return _Resp({"status": "ready"})
        return _Resp({"detail": "n/a"}, 503)

    monkeypatch.setattr(httpx, "get", fake_get)
    monkeypatch.setattr(httpx, "post", lambda url, **_: _Resp(body))
    at = AppTest.from_file(DASHBOARD, default_timeout=30).run()
    assert not at.exception
    assert any("Local LLM unavailable" in w.value for w in at.warning)
    next(b for b in at.button if b.label == "Run research").click()
    at.run()
    assert not at.exception
    text = " ".join(x.value for x in [*at.markdown, *at.error, *at.warning, *at.subheader])
    assert "Verified historical facts" in text and "no demonstrated predictive edge" in text
    assert "SYNTHETIC DEMONSTRATION" in text and "investment advice" in text


def test_agent_tab_shows_api_error(monkeypatch: pytest.MonkeyPatch) -> None:
    from streamlit.testing.v1 import AppTest

    monkeypatch.setattr(
        httpx,
        "get",
        lambda url, **_: _Resp({"status": "ready"}) if "health" in url else _Resp({}, 503),
    )
    monkeypatch.setattr(httpx, "post", lambda url, **_: _Resp({"detail": "Broker disabled"}, 403))
    at = AppTest.from_file(DASHBOARD, default_timeout=30).run()
    next(b for b in at.button if b.label == "Run research").click()
    at.run()
    assert any("Broker disabled" in e.value for e in at.error)
