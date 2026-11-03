"""Deterministic metric catalogue, evidence lookup and structured comparisons.

The LLM may *propose* a comparison as structured data (metric, two subjects, the evidence ids and
a claimed relation). Code resolves both values from the cited evidence, checks that they are
comparable (same tool, evaluation period, window and unit), computes the real relation and writes
the sentence itself. A proposal that cannot be established is rejected with a reason.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal

from marketops.agents.schemas import Claim, Evidence, RejectedClaim

Unit = Literal["pct", "ratio"]
MetricClass = Literal["classification", "strategy", "price"]
Relation = Literal["higher", "lower", "approximately_equal"]


@dataclass(frozen=True)
class Metric:
    key: str  # exact key used in tool payloads
    label: str
    pattern: str  # regex over lower-cased text
    cls: MetricClass
    unit: Unit
    lower_is_better: bool | None = None
    aliases: tuple[str, ...] = ()  # other payload keys holding the same quantity

    @property
    def keys(self) -> tuple[str, ...]:
        return (self.key, *self.aliases)


METRICS: tuple[Metric, ...] = (
    Metric(
        "roc_auc",
        "ROC-AUC",
        r"roc[-_ ]?auc|auc[-_ ]?roc|\bauc\b",
        "classification",
        "ratio",
        aliases=("oof_roc_auc", "fold_auc_mean", "fold_auc_min"),
    ),
    Metric("pr_auc", "PR-AUC", r"pr[-_ ]?auc|average precision", "classification", "ratio"),
    Metric("brier", "Brier score", r"brier", "classification", "ratio", True, ("oof_brier",)),
    Metric(
        "log_loss", "log-loss", r"log[-_ ]?loss", "classification", "ratio", True, ("oof_log_loss",)
    ),
    Metric("f1", "F1", r"\bf1\b", "classification", "ratio"),
    Metric(
        "balanced_accuracy",
        "balanced accuracy",
        r"balanced[-_ ]accuracy",
        "classification",
        "ratio",
    ),
    # Plain "accuracy"/"precision"/"recall" are recognised so they are not silently skipped;
    # no tool returns them under those names, so a number bound to them is never supported.
    Metric("accuracy", "accuracy", r"\baccuracy\b", "classification", "ratio"),
    Metric("precision", "precision", r"\bprecision\b", "classification", "ratio"),
    Metric("recall", "recall", r"\brecall\b", "classification", "ratio"),
    Metric("cagr", "CAGR", r"\bcagr\b|compound annual", "strategy", "pct"),
    Metric("sharpe", "Sharpe ratio", r"sharpe", "strategy", "ratio"),
    Metric(
        "total_return",
        "total return",
        r"total[-_ ]return",
        "strategy",
        "pct",
        aliases=("window_total_return_adj",),  # price evidence calls it the window return
    ),
    Metric("ann_vol", "annualised volatility", r"volatility|ann_vol", "strategy", "pct", True),
    Metric("max_drawdown", "max drawdown", r"drawdown", "strategy", "pct"),
    Metric("annual_turnover", "annual turnover", r"turnover", "strategy", "ratio", True),
    Metric(
        "window_total_return_adj",
        "adjusted window return",
        r"window return|adjusted (?:window )?return|window_total_return_adj",
        "price",
        "pct",
    ),
)
BY_KEY = {m.key: m for m in METRICS}
_METRIC_RE = re.compile(
    "|".join(f"(?P<{m.key}>{m.pattern})" for m in METRICS),
    re.I,
)

# Subjects inside backtest / evaluation payloads: canonical payload key -> text aliases.
SUBJECTS: dict[str, str] = {
    "model": r"\bml\b|\bmodel(?:'s)?\b|\blogreg\b|\bclassifier\b",
    "momentum_126d": r"momentum[- _]?126\w*",
    "baseline_momentum_21d": r"momentum[- _]?21\w*",
    "momentum": r"momentum(?![- _]?\d)",
    "model_nonoverlap_subsample": r"non-?overlap\w*",
    "variant": r"logreg_c[\d.]*\d|xgb_d\d+_n\d+",
    "equal_weight": r"equal[- _]weight",
    "spy_buy_hold": r"\bspy\b|spy_buy_hold|s&p|buy[-_ ]and[-_ ]hold",
    "baseline_always_up": r"always[- _]up|always predict\w*",
}
_SUBJECT_RE = re.compile("|".join(f"(?P<{k}>{v})" for k, v in SUBJECTS.items()), re.I)
# Evaluation payloads store the always-up baseline under baseline_always_up, backtests under others.
_PERIOD_TEST = re.compile(r"held[- ]out|out[- ]of[- ]sample|test[- ](?:period|set|window)", re.I)
_PERIOD_DEV = re.compile(
    r"walk[- ]forward|cross[- ]validat|in[- ]sample|development|\bdev\b|\boof\b", re.I
)


def metric_in(text: str) -> list[tuple[Metric, int, int]]:
    out = []
    for m in _METRIC_RE.finditer(text):
        key = m.lastgroup
        if key:
            out.append((BY_KEY[key], m.start(), m.end()))
    return out


# A bare "momentum" names the momentum strategy of whichever tool is cited (126d / 21d).
SUBJECT_KEYS: dict[str, tuple[str, ...]] = {"momentum": ("momentum_126d", "baseline_momentum_21d")}


def _keys(subject: str) -> tuple[str, ...]:
    return SUBJECT_KEYS.get(subject, (subject,))


def subject_in(text: str) -> list[tuple[str, int, int]]:
    out = []
    for m in _SUBJECT_RE.finditer(text):
        key = m.group().lower() if m.lastgroup == "variant" else (m.lastgroup or "")
        out.append((key, m.start(), m.end()))
    return out


def strip_subjects(text: str) -> str:
    """Blank out subject names so they do not count as foreign words between metric and value."""
    return _SUBJECT_RE.sub(" ", text)


def subject_matches(subject: str, path: tuple[str, ...]) -> bool:
    wanted = {k.lower() for k in _keys(subject)}
    return any(p.lower() in wanted for p in path)


def period_words(text: str) -> set[str]:
    """Evaluation period the prose refers to: 'test' (held-out) and/or 'dev'."""
    out = set()
    if _PERIOD_TEST.search(text):
        out.add("test")
    if _PERIOD_DEV.search(text):
        out.add("dev")
    return out


@dataclass(frozen=True)
class Leaf:
    evidence: str
    tool: str
    path: tuple[str, ...]
    value: float
    period: str | None  # 'test' | 'dev' | None
    symbol: str | None


def _walk(obj: Any, path: tuple[str, ...]) -> list[tuple[tuple[str, ...], float]]:
    if isinstance(obj, bool):
        return []
    if isinstance(obj, (int, float)):
        return [(path, float(obj))]
    out: list[tuple[tuple[str, ...], float]] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            out += _walk(v, (*path, str(k)))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out += _walk(v, (*path, str(i)))
    return out


def leaves(ev: Evidence) -> list[Leaf]:
    """Numeric leaves of one evidence payload, tagged with the evaluation period they belong to."""
    if not ev.data:
        return []
    bt_period = None
    if ev.tool == "get_backtest":
        bt_period = "test" if ev.data.get("period") == "test" else "dev"
    symbol = ev.data.get("symbol") if isinstance(ev.data.get("symbol"), str) else None
    out = []
    for path, v in _walk(ev.data, ()):
        period = bt_period
        if "held_out_test" in path:
            period = "test"
        elif "walk_forward_cv" in path:
            period = "dev"
        out.append(Leaf(ev.id, ev.tool, path, v, period, symbol))
    return out


def fmt(metric: Metric, v: float) -> str:
    if metric.unit == "pct":
        return f"{v * 100:.2f}%"
    return f"{v:.2f}" if metric.key in ("sharpe", "annual_turnover") else f"{v:.4f}"


def resolve(ev: Evidence, metric: Metric, subject: str) -> float | None:
    """Value of `metric` for `subject` in one evidence payload, or None."""
    d = ev.data or {}
    v: Any = None
    if ev.tool == "get_price_summary" and metric.key == "window_total_return_adj":
        v = d.get(metric.key) if str(d.get("symbol", "")).upper() == subject.upper() else None
    elif ev.tool == "get_backtest":
        table = d.get("metrics") or {}
        v = next(
            (table[k][metric.key] for k in _keys(subject) if metric.key in table.get(k, {})), None
        )
    elif ev.tool == "get_model_evaluation":
        table = d.get("held_out_test") or {}
        v = next(
            (table[k][metric.key] for k in _keys(subject) if metric.key in table.get(k, {})), None
        )
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def context(ev: Evidence) -> tuple[Any, ...]:
    """What must be identical for two values to be comparable."""
    d = ev.data or {}
    if ev.tool == "get_price_summary":
        return (ev.tool, d.get("first_date"), d.get("last_date"), d.get("bars"))
    if ev.tool == "get_backtest":
        return (ev.tool, d.get("period"), d.get("period_start"), d.get("period_end"))
    return (ev.tool, d.get("model_version"))


def describe(ev: Evidence) -> str:
    d = ev.data or {}
    if ev.tool == "get_price_summary":
        return (
            f"adjusted window {d.get('first_date')} to {d.get('last_date')} ({d.get('bars')} bars)"
        )
    if ev.tool == "get_backtest":
        span = f" {d.get('period_start')} to {d.get('period_end')}" if d.get("period_start") else ""
        name = "held-out test" if d.get("period") == "test" else "development out-of-fold"
        return f"historical {name} backtest{span}"
    return "held-out test evaluation"


def _canon_metric(name: str) -> Metric | None:
    found = metric_in(name.lower())
    return found[0][0] if len(found) == 1 else None


def _canon_subject(name: str, symbols: set[str]) -> str | None:
    if name.upper() in symbols:
        return name.upper()
    found = {k for k, _, _ in subject_in(name.lower())}
    return next(iter(found)) if len(found) == 1 else None


def evaluate_comparison(
    metric_name: str,
    subject_a: str,
    evidence_a: str,
    subject_b: str,
    evidence_b: str,
    relation: Relation,
    constant: float | None,
    by_id: dict[str, Evidence],
) -> tuple[Claim | None, str | None]:
    """Return (claim, None) when established by evidence, else (None, rejection reason)."""
    symbols = {
        str(e.data["symbol"]).upper() for e in by_id.values() if e.data and "symbol" in e.data
    }
    metric = _canon_metric(metric_name)
    if metric is None:
        return None, f"unknown or ambiguous metric {metric_name[:40]!r}"
    a = _canon_subject(subject_a, symbols)
    b = _canon_subject(subject_b, symbols) if constant is None else "constant"
    if a is None or b is None:
        return None, "subject is not a recognised strategy, model, baseline or symbol"
    if a == b:
        return None, "both sides of the comparison are the same subject"
    ea = by_id.get(evidence_a)
    eb = by_id.get(evidence_b) if constant is None else ea
    if ea is None or eb is None:
        return None, "comparison cites evidence that was not retrieved in this run"
    if ea.sensitive or eb.sensitive:
        return None, "comparisons on broker data are not produced"
    va = resolve(ea, metric, a)
    if va is None:
        return None, f"{metric.label} for {a} is not in {ea.id}"
    if constant is None:
        vb = resolve(eb, metric, b)
        if vb is None:
            return None, f"{metric.label} for {b} is not in {eb.id}"
        if context(ea) != context(eb):
            return None, (
                f"values come from different evaluation windows or tools "
                f"({describe(ea)} vs {describe(eb)})"
            )
        right = f"{b} {fmt(metric, vb)}"
    else:
        vb = float(constant)
        right = f"the constant {constant:g}"
    tol = 0.5 * 10**-4 if metric.unit == "ratio" else 0.5 * 10**-4
    actual: Relation = (
        "approximately_equal" if abs(va - vb) <= tol else ("higher" if va > vb else "lower")
    )
    if actual != relation:
        return None, (
            f"claimed {a} is {relation.replace('_', ' ')} but evidence gives "
            f"{fmt(metric, va)} vs {fmt(metric, vb)} ({actual.replace('_', ' ')})"
        )
    verb = "approximately equal to" if actual == "approximately_equal" else f"{actual} than"
    note = (
        " Lower is the more favourable direction for this metric."
        if metric.lower_is_better and actual != "approximately_equal"
        else ""
    )
    text = (
        f"Computed from {ea.id}{'' if eb is ea else ' and ' + eb.id}: {a} {metric.label} "
        f"{fmt(metric, va)} is {verb} {right} over {describe(ea)}. Point estimates from "
        f"one historical sample; no significance test was run.{note}"
    )
    ids = [ea.id] if eb is ea else [ea.id, eb.id]
    return Claim(text=text, evidence=ids), None


def evaluate_all(
    proposals: list[Any], by_id: dict[str, Evidence]
) -> tuple[list[Claim], list[RejectedClaim]]:
    ok: list[Claim] = []
    bad: list[RejectedClaim] = []
    for p in proposals[:6]:
        claim, reason = evaluate_comparison(
            p.metric,
            p.subject_a,
            p.evidence_a,
            p.subject_b,
            p.evidence_b,
            p.relation,
            p.constant,
            by_id,
        )
        if claim is not None:
            ok.append(claim)
        else:
            summary = f"{p.subject_a} {p.relation} {p.metric} {p.subject_b or p.constant}"
            bad.append(RejectedClaim(section="comparisons", text=summary[:300], reason=str(reason)))
    for p in proposals[6:]:
        bad.append(
            RejectedClaim(
                section="comparisons",
                text=f"{p.subject_a} vs {p.subject_b} {p.metric}"[:300],
                reason="comparison limit (6) exceeded",
            )
        )
    return ok, bad
