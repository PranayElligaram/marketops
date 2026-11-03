"""Claim-verification regressions: false comparisons, metric/unit/period binding, forward-looking
backtests, classification-vs-return confusion. Cases marked LIVE were accepted by the previous
verifier in a real qwen3.5:4b run on 2026-10-28 (artefact 20261028T221802Z)."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import pytest

from marketops.agents import compare
from marketops.agents.runner import _Comparison
from marketops.agents.schemas import Claim, Evidence
from marketops.agents.tools import Toolbox
from marketops.agents.verify import check_claim, filter_notes
from marketops.config import Settings
from tests.test_agent import settings  # noqa: F401  (fixture reuse)


def _ev(tb: Toolbox, eid: str, tool: str, **args: Any) -> Evidence:
    tdef, vargs, data, source = tb.execute(tool, args)
    return Evidence(
        id=eid,
        tool=tool,
        args=vargs,
        kind=tdef.kind,
        source=source,
        retrieved_utc="t",
        sha256=hashlib.sha256(json.dumps(data).encode()).hexdigest(),
        data=data,
    )


@pytest.fixture
def ev(settings: Settings) -> dict[str, Evidence]:  # noqa: F811
    tb = Toolbox(settings)
    items = [
        _ev(tb, "E1", "get_model_evaluation"),
        _ev(tb, "E2", "get_backtest", period="test"),
        _ev(tb, "E3", "get_backtest", period="dev_oof"),
        _ev(tb, "E4", "get_price_summary", symbol="AAA", days=60),
        _ev(tb, "E5", "get_price_summary", symbol="BBB", days=60),
        _ev(tb, "E6", "get_price_summary", symbol="BBB", days=100),
    ]
    return {e.id: e for e in items}


def reason(ev: dict[str, Evidence], text: str, *cites: str, section: str = "verified_facts") -> Any:
    return check_claim(section, Claim(text=text, evidence=list(cites)), ev)


# ------------------------------------------------------------------ prose comparisons
@pytest.mark.parametrize(
    "text",
    [
        # LIVE: false (0.4961 > 0.4944) yet accepted before.
        "The ML model's classification metrics (ROC-AUC: 0.4961, F1: 0.6086) were lower than the "
        "momentum baseline (ROC-AUC: 0.4944, F1: 0.561).",
        # LIVE: -16.03% is a shallower drawdown than -17.39%; "lower" is ambiguous and was accepted.
        "The model had a slightly lower maximum drawdown (-16.03%) compared to equal-weight "
        "(-17.39%).",
        "The model outperformed SPY on a historical backtest.",
        "The model's Sharpe ratio was better than the equal-weight strategy.",
    ],
)
def test_prose_comparisons_are_rejected(ev: dict[str, Evidence], text: str) -> None:
    assert "comparison" in str(reason(ev, text, "E1", "E2"))


# ------------------------------------------------------------------ structured comparisons
def _cmp(ev: dict[str, Evidence], **kw: Any) -> tuple[Claim | None, str | None]:
    base: dict[str, Any] = {
        "metric": "cagr",
        "subject_a": "model",
        "evidence_a": "E2",
        "subject_b": "equal_weight",
        "evidence_b": "E2",
        "relation": "lower",
        "constant": None,
    }
    base.update(kw)
    return compare.evaluate_comparison(
        base["metric"],
        base["subject_a"],
        base["evidence_a"],
        base["subject_b"],
        base["evidence_b"],
        base["relation"],
        base["constant"],
        ev,
    )


def test_supported_comparison_is_generated_by_code(ev: dict[str, Evidence]) -> None:
    claim, why = _cmp(ev)
    assert why is None and claim is not None
    assert "14.39%" in claim.text and "17.29%" in claim.text and "lower than" in claim.text
    assert "2022-01-03 to 2026-10-26" in claim.text and "no significance test" in claim.text
    assert claim.evidence == ["E2"]


@pytest.mark.parametrize(
    ("kw", "needle"),
    [
        ({"relation": "higher"}, "evidence gives"),  # wrong direction
        ({"relation": "approximately_equal"}, "evidence gives"),
        ({"evidence_b": "E3", "subject_b": "equal_weight"}, "different evaluation"),  # periods
        ({"subject_b": "model"}, "same subject"),
        ({"subject_b": "nasdaq"}, "not a recognised"),
        ({"metric": "alpha"}, "unknown or ambiguous"),
        ({"metric": "roc_auc"}, "not in E2"),  # classification metric vs backtest evidence
        ({"evidence_a": "E99"}, "not retrieved"),
        ({"subject_b": "momentum_126d", "evidence_b": "E3"}, "not in E3"),  # absent from dev_oof
    ],
)
def test_unestablished_comparisons_are_rejected(
    ev: dict[str, Evidence], kw: dict[str, Any], needle: str
) -> None:
    claim, why = _cmp(ev, **kw)
    assert claim is None and needle in str(why)


def test_live_roc_auc_claim_fixed_as_structured_comparison(ev: dict[str, Evidence]) -> None:
    # model 0.4961 vs always-up 0.5 on the fixture: "higher" is false, "lower" is true.
    bad, why = _cmp(
        ev,
        metric="roc_auc",
        evidence_a="E1",
        evidence_b="E1",
        subject_b="baseline_always_up",
        relation="higher",
    )
    assert bad is None and "evidence gives" in str(why)
    ok, _ = _cmp(
        ev,
        metric="roc_auc",
        evidence_a="E1",
        evidence_b="E1",
        subject_b="baseline_always_up",
        relation="lower",
    )
    assert ok is not None and "held-out test evaluation" in ok.text


def test_constant_comparison_against_chance(ev: dict[str, Evidence]) -> None:
    ok, _ = _cmp(
        ev, metric="roc_auc", evidence_a="E1", subject_b="", constant=0.5, relation="lower"
    )
    assert ok is not None and "the constant 0.5" in ok.text
    bad, _ = _cmp(
        ev, metric="roc_auc", evidence_a="E1", subject_b="", constant=0.5, relation="higher"
    )
    assert bad is None


def test_symbol_comparison_requires_identical_windows(ev: dict[str, Evidence]) -> None:
    kw = {"metric": "window return", "subject_a": "AAA", "subject_b": "BBB", "evidence_a": "E4"}
    mismatch, why = _cmp(ev, evidence_b="E6", relation="higher", **kw)
    assert mismatch is None and "different evaluation windows" in str(why)
    a = ev["E4"].data["window_total_return_adj"]  # type: ignore[index]
    b = ev["E5"].data["window_total_return_adj"]  # type: ignore[index]
    right = "higher" if a > b else "lower"
    ok, _ = _cmp(ev, evidence_b="E5", relation=right, **kw)
    assert ok is not None and "adjusted window" in ok.text
    wrong = "lower" if right == "higher" else "higher"
    assert _cmp(ev, evidence_b="E5", relation=wrong, **kw)[0] is None


def test_equal_values_and_lower_is_better_note(ev: dict[str, Evidence]) -> None:
    same, _ = _cmp(
        ev,
        metric="roc_auc",
        evidence_a="E1",
        subject_b="",
        constant=0.4961,
        relation="approximately_equal",
    )
    assert same is not None and "approximately equal to" in same.text
    # model Brier 0.2483 vs always-up 0.2477: numerically higher, i.e. the less favourable side
    brier, _ = _cmp(
        ev,
        metric="brier",
        evidence_a="E1",
        evidence_b="E1",
        subject_b="baseline_always_up",
        relation="higher",
    )
    assert brier is not None and "Lower is the more favourable" in brier.text
    assert (
        _cmp(
            ev,
            metric="brier",
            evidence_a="E1",
            evidence_b="E1",
            subject_b="baseline_always_up",
            relation="approximately_equal",
        )[0]
        is None
    )


def test_comparison_proposals_flow_through_report(settings: Settings) -> None:  # noqa: F811
    from tests.test_agent import Fake, calls, run_research, synthesis, tc

    fake = Fake(
        calls(tc("get_backtest", period="test")),
        synthesis(
            comparisons=[
                {
                    "metric": "sharpe",
                    "subject_a": "model",
                    "evidence_a": "E2",
                    "subject_b": "equal_weight",
                    "evidence_b": "E2",
                    "relation": "lower",
                },
                {
                    "metric": "sharpe",
                    "subject_a": "model",
                    "evidence_a": "E2",
                    "subject_b": "spy_buy_hold",
                    "evidence_b": "E2",
                    "relation": "lower",  # false: 0.92 > 0.74
                },
            ]
        ),
    )
    report, _ = run_research("backtest?", "none", settings, fake)
    assert any("Sharpe ratio 0.92 is lower than" in c.text for c in report.verified_facts)
    rej = [r for r in report.rejected_claims if r.section == "comparisons"]
    assert len(rej) == 1 and "evidence gives" in rej[0].reason
    assert report.mode == "llm"


def test_comparison_schema_rejects_unknown_relation() -> None:
    with pytest.raises(ValueError, match="relation"):
        _Comparison(metric="cagr", subject_a="a", evidence_a="E1", relation="better")  # type: ignore[arg-type]


# ------------------------------------------------------------------ metric / unit / period binding
@pytest.mark.parametrize(
    ("text", "cites", "needle"),
    [
        ("Historical backtest: the model's ROC-AUC was 0.9243.", ("E2",), "not in the cited"),
        ("The model's held-out ROC-AUC was 0.9243.", ("E1",), "does not match"),  # a Sharpe value
        ("Historical backtest Sharpe of the model was 92.4%.", ("E2",), "not a percentage"),
        ("Historical backtest: equal-weight CAGR was 14.39%.", ("E2",), "does not match"),
        ("In the held-out test, the model's CAGR was 14.39%.", ("E3",), "period"),
        ("In walk-forward development, the model's CAGR was 14.39%.", ("E2",), "period"),
        ("BBB's window return was " + "{AAA_RET}" + ".", ("E4", "E5"), "does not match"),
        ("Historical backtest: the model's accuracy was 0.55.", ("E2",), "not in the cited"),
    ],
)
def test_bound_numbers_must_match_metric_subject_period(
    ev: dict[str, Evidence], text: str, cites: tuple[str, ...], needle: str
) -> None:
    text = text.replace("{AAA_RET}", f"{ev['E4'].data['window_total_return_adj']:.4f}")  # type: ignore[index]
    assert needle in str(reason(ev, text, *cites))


@pytest.mark.parametrize(
    ("text", "cites"),
    [
        ("Historical backtest: the model's CAGR was 14.39%.", ("E2",)),
        ("Historical backtest, model CAGR 0.1439 and Sharpe 0.92.", ("E2",)),
        ("Historical backtest: equal-weight CAGR was 17.29% and SPY CAGR 11.86%.", ("E2",)),
        ("The held-out ROC-AUC of the model was 0.4961.", ("E1",)),
        ("Held-out Brier score of the always-up baseline was 0.2477.", ("E1",)),
        ("Held-out test: the always-up baseline ROC-AUC was 0.5.", ("E1",)),
        ("Walk-forward logreg_C0.01 ROC-AUC mean was 0.5166.", ("E1",)),
    ],
)
def test_correctly_bound_claims_pass(
    ev: dict[str, Evidence], text: str, cites: tuple[str, ...]
) -> None:
    assert reason(ev, text, *cites) is None


# ---------------------------------------------- forward-looking / classification
@pytest.mark.parametrize(
    "text",
    [
        "The model's historical backtest CAGR of 14.39% is expected to continue.",
        "Historical backtest results suggest the model will deliver a CAGR of 14.39%.",
        "Going forward the backtest Sharpe of 0.92 should repeat.",
        "Backtest CAGR of 14.39% is a projection of future returns.",
    ],
)
def test_backtests_not_presented_as_future_performance(ev: dict[str, Evidence], text: str) -> None:
    assert "future performance" in str(reason(ev, text, "E2"))


def test_unlabelled_backtest_numbers_rejected(ev: dict[str, Evidence]) -> None:
    assert "historical" in str(reason(ev, "The model's CAGR is 14.39%.", "E2"))


def test_classification_metrics_not_linked_to_returns(ev: dict[str, Evidence]) -> None:
    text = "The model's held-out ROC-AUC of 0.4961 explains its historical backtest CAGR of 14.39%."
    assert "classification performance to strategy returns" in str(reason(ev, text, "E1", "E2"))
    alone = "A held-out ROC-AUC of 0.4961 is a classification result, not a return."
    assert reason(ev, alone, "E1") is None


def test_uncertainty_notes_are_verified_too(ev: dict[str, Evidence]) -> None:
    # LIVE: the model wrote a wrong remark about a "synthetic universe" in an earlier run.
    notes = [
        "The universe is synthetic, so results may not generalise.",
        "No fundamentals were retrieved.",
        "The model's Sharpe ratio was better than SPY.",
    ]
    kept, rejected = filter_notes(notes, ev)
    assert kept == ["No fundamentals were retrieved."]
    assert {r.section for r in rejected} == {"missing_or_uncertain"} and len(rejected) == 2
    kept2, _ = filter_notes(notes[:1], ev, allow_synthetic=True)
    assert kept2 == notes[:1]


# --------------------------------------------- live false positives (2026-10-28)
def test_live_unrelated_number_after_metric_is_not_misbound(ev: dict[str, Evidence]) -> None:
    # "0.5493" is the base rate; it was wrongly bound to the preceding balanced accuracy.
    text = "Held-out balanced accuracy was 0.4916 and the base rate was 0.5493."
    ev["E1"].data["held_out_test"]["model"]["balanced_accuracy"] = 0.4916  # type: ignore[index]
    ev["E1"].data["held_out_test"]["model"]["base_rate"] = 0.5493  # type: ignore[index]
    assert reason(ev, text, "E1") is None


def test_live_total_return_on_price_evidence(ev: dict[str, Evidence]) -> None:
    v = ev["E4"].data["window_total_return_adj"]  # type: ignore[index]
    assert reason(ev, f"AAA total return over the window was {v:.4f}.", "E4") is None
    assert reason(ev, "AAA total return over the window was 0.9999.", "E4") is not None
