"""Gate: the deterministic agent evaluation set must pass completely (docs/agent-evaluation.md)."""

from __future__ import annotations

from tests.agent_eval.harness import evaluate


def test_agent_evaluation_set_passes() -> None:
    outcomes, m = evaluate()
    failures = {o.case.id: o.problems for o in outcomes if o.problems}
    assert not failures, failures
    assert m["cases"] >= 25
    assert m["claim_acceptance"] == 1.0
    assert m["unsupported_rejection"] == 1.0 and m["false_accepts"] == 0
    assert m["tool_call_correctness"] == 1.0


def test_every_category_is_covered() -> None:
    outcomes, _ = evaluate()
    cats = {o.case.category for o in outcomes}
    assert cats >= {
        "factual",
        "supported_comparison",
        "unsupported",
        "missing_data",
        "adversarial",
        "portfolio_privacy",
    }
