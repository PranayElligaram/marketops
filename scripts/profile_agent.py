"""Profile real local-Ollama research runs: time in LLM generation, tools, verification.

Usage: python scripts/profile_agent.py OUT.json [--repeat N]
Wraps the HTTP layer, Toolbox.execute and the verifier from outside, so it measures any version of
the agent. Needs a running Ollama with the configured model; makes no other network calls.
"""

from __future__ import annotations

import json
import sys
import time
from typing import Any

import httpx

from marketops.agents import runner, tools
from marketops.agents.ollama import OllamaClient
from marketops.config import get_settings

QUESTIONS = {
    "price_signal": "Summarise AAPL's recent price history and the model's current signal.",
    "backtest_compare": (
        "How did the ML strategy compare with the equal-weight and SPY baselines in the backtest?"
    ),
    "edge": (
        "Does the model's held-out evaluation show a predictive edge over always predicting up?"
    ),
}


def main() -> None:
    out_path = sys.argv[1]
    repeat = int(sys.argv[sys.argv.index("--repeat") + 1]) if "--repeat" in sys.argv else 1
    s = get_settings()
    client = OllamaClient(s.ollama_base_url, s.ollama_model, timeout=s.agent_timeout_s)
    rec: dict[str, Any] = {"llm": [], "tools_ms": 0.0, "verify_ms": 0.0}

    orig_req = client._request

    def req(method: str, path: str, timeout: float, **kw: Any) -> httpx.Response:
        t0 = time.perf_counter()
        r = orig_req(method, path, timeout, **kw)
        if path == "/api/chat" and r.status_code == 200:
            j = r.json()
            rec["llm"].append(
                {
                    "kind": "synthesis" if "format" in kw["json"] else "tool_phase",
                    "wall_ms": round((time.perf_counter() - t0) * 1000),
                    "prompt_tokens": j.get("prompt_eval_count"),
                    "output_tokens": j.get("eval_count"),
                    "load_ms": round(j.get("load_duration", 0) / 1e6),
                }
            )
        return r

    client._request = req  # type: ignore[method-assign]

    orig_exec = tools.Toolbox.execute

    def ex(self: tools.Toolbox, name: Any, raw: Any) -> Any:
        t0 = time.perf_counter()
        try:
            return orig_exec(self, name, raw)
        finally:
            rec["tools_ms"] += (time.perf_counter() - t0) * 1000

    tools.Toolbox.execute = ex  # type: ignore[method-assign]

    orig_filter = runner.filter_claims

    def fc(*a: Any, **k: Any) -> Any:
        t0 = time.perf_counter()
        try:
            return orig_filter(*a, **k)
        finally:
            rec["verify_ms"] += (time.perf_counter() - t0) * 1000

    runner.filter_claims = fc

    results = []
    for name, q in QUESTIONS.items():
        for i in range(repeat):
            rec.update(llm=[], tools_ms=0.0, verify_ms=0.0)
            t0 = time.perf_counter()
            rep, _ = runner.run_research(q, "none", s, client)
            wall = (time.perf_counter() - t0) * 1000
            llm_ms = sum(c["wall_ms"] for c in rec["llm"])
            row = {
                "question": name,
                "rep": i,
                "wall_ms": round(wall),
                "llm_ms": llm_ms,
                "llm_tool_phase_ms": sum(
                    c["wall_ms"] for c in rec["llm"] if c["kind"] == "tool_phase"
                ),
                "llm_synthesis_ms": sum(
                    c["wall_ms"] for c in rec["llm"] if c["kind"] == "synthesis"
                ),
                "tools_ms": round(rec["tools_ms"]),
                "verify_ms": round(rec["verify_ms"], 1),
                "other_ms": round(wall - llm_ms - rec["tools_ms"] - rec["verify_ms"]),
                "llm_calls": len(rec["llm"]),
                "prompt_tokens": sum(c["prompt_tokens"] or 0 for c in rec["llm"]),
                "output_tokens": sum(c["output_tokens"] or 0 for c in rec["llm"]),
                "tool_calls": rep.tool_calls,
                "mode": rep.mode,
                "llm_status": rep.llm_status,
                "status": rep.status,
                "facts": len(rep.verified_facts),
                "estimates": len(rep.model_estimates),
                "interpretation": len(rep.interpretation),
                "rejected": len(rep.rejected_claims),
                "rejected_reasons": [r.reason for r in rep.rejected_claims],
                "tools": [t.tool for t in rep.tool_trace],
                "evidence_chars": sum(len(json.dumps(e.data)) for e in rep.evidence),
                "claims": [
                    c.text for c in rep.verified_facts + rep.model_estimates + rep.interpretation
                ],
            }
            results.append(row)
            print(json.dumps({k: v for k, v in row.items() if k != "claims"}), flush=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
