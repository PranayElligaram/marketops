"""Deterministic checks on LLM-written claims: citations, numbers, metrics, periods, comparisons."""

from __future__ import annotations

import json
import re
from typing import Any

from marketops.agents import compare
from marketops.agents.schemas import Claim, Evidence, RejectedClaim

MAX_CLAIM_CHARS = 500
_EID = re.compile(r"\bE\d+\b")
_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
_NUM = re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)?")
_LINKS = re.compile(r"https?://|!\[|<\s*(script|img|iframe)|javascript:", re.I)
_ADVICE = re.compile(
    r"\b(you should|i recommend|we recommend|recommend(?:ed)? (?:to )?(?:buy|sell|hold)|"
    r"(?:strong )?(?:buy|sell) (?:signal|rating|recommendation)|guaranteed|risk[- ]free|"
    r"(?:will|is going to) (?:rise|increase|go up|outperform|beat|fall|drop)|"
    r"can't lose|cannot lose|place an? (?:order|trade)|significantly)\b",
    re.I,
)
_EDGE = re.compile(
    r"(?<!no )(?<!not )(?<!without )(?<!lacks )\b"
    r"(?:proven|reliable|strong|demonstrated|significant|"
    r"statistically significant|good)\s+(?:predictive\s+)?(?:edge|power|accuracy|skill|alpha)\b",
    re.I,
)
# Prose comparisons cannot be checked reliably, so they must arrive as structured comparisons that
# code evaluates (compare.py). This is a fail-closed guard on top of, not instead of, the metric,
# unit, subject and period binding below.
_COMPARE = re.compile(
    r"\b(?:higher|lower|greater|larger|smaller|bigger|more|less|better|worse|stronger|weaker|"
    r"superior|inferior|outperform\w*|underperform\w*|beat|beats|beating|exceed\w*|surpass\w*|"
    r"trail\w*|lag\w*|ahead|behind|above|below|versus|vs|than|compared|relative to|"
    r"improv\w*|declin\w*|increas\w*|decreas\w*|wins?|loses?|rose|fell)\b",
    re.I,
)
_FORWARD = re.compile(
    r"\b(?:expect\w*|future|going forward|going to|likely to|projected|projection|forecast\w*|"
    r"anticipat\w*|will (?!not\b)\w+|should (?:earn|deliver|return|continue|repeat|yield|"
    r"generate)|next (?:year|month|quarter)|long[- ]run|over time)\b",
    re.I,
)
_HISTORICAL = re.compile(
    r"\b(?:historical\w*|backtest\w*|back-test\w*|simulat\w*|held-out|test period|"
    r"out-of-sample|walk-forward|in-sample|during|between|from \d{4})\b",
    re.I,
)
_CONNECTOR = re.compile(
    r"\b(?:because|therefore|thus|hence|so that|implies|implying|imply|explains?|translates?|"
    r"leads? to|results? in|driv\w+|due to|consequently|which means|means that|indicates?)\b",
    re.I,
)
_RETURN_WORDS = re.compile(r"\b(?:returns?|profit\w*|gains?|earn\w*|alpha|pnl|p&l)\b", re.I)
_SYNTHETIC = re.compile(r"\b(?:synthetic|simulated data|fake|placeholder|mock|dummy|demo)\b", re.I)
_TICKER = re.compile(r"\b[A-Z][A-Z0-9.\-]{0,9}\b")
INJECTION = re.compile(
    r"(ignore|disregard|forget)\s+(all\s+|any\s+)?(previous|prior|above|earlier)|"
    r"(reveal|print|show)\s+(your\s+)?(system\s+)?prompt|you are now|new instructions|"
    r"developer mode|jailbreak|act as (?:an? )?(?:unrestricted|dan)",
    re.I,
)
# Words that may sit between a metric name and its value ("Sharpe was about 0.92"). Anything else
# ("Brier 0.2483, base rate 0.5493") means the number belongs to another quantity: not bound.
_GAP = re.compile(
    r"^[\s:=(,\-]*(?:(?:of|was|is|were|are|at|to|score|ratio|value|mean|about|approximately|"
    r"around|near|equal|equals|measured|reached|had|has|stood|in|the|a|and|for)\b[\s:=(,\-]*)*$"
)
_MINUS = "-" + chr(0x2212)


def has_injection(text: str) -> bool:
    return bool(INJECTION.search(text))


def _numbers(obj: Any) -> list[float]:
    text = json.dumps(obj, default=str)
    return [abs(float(m.replace(",", ""))) for m in _NUM.findall(text)]


def _matches(value: float, decimals: int, pool: list[float]) -> bool:
    tol = 0.5 * 10**-decimals + 1e-9
    return any(abs(c - value) <= tol or abs(c * 100 - value) <= tol for c in pool)


def _blank(m: re.Match[str]) -> str:
    return " " * len(m.group())


def bound_metric_problems(text: str, evidence: list[Evidence]) -> list[str]:
    """Check each number next to a metric name against that metric's own values.

    A number binds to the nearest preceding metric keyword in the same clause (or one directly
    after it, e.g. "14.4% CAGR"). The metric must exist in the cited evidence, the unit must fit,
    and when the clause names a strategy/model/symbol or an evaluation period the value must come
    from that subject and period.
    """
    clean_text = _DATE.sub(_blank, _EID.sub(_blank, text))
    low = clean_text.lower()
    metrics = compare.metric_in(low)
    if not metrics:
        return []
    subjects = compare.subject_in(low)
    periods = compare.period_words(low)
    symbols = {str(e.data["symbol"]).upper() for e in evidence if e.data and e.data.get("symbol")}
    named = {w for w in _TICKER.findall(clean_text) if w in symbols}
    all_leaves = [lf for e in evidence for lf in compare.leaves(e)]
    problems: list[str] = []
    for nm in _NUM.finditer(low):
        tok = nm.group()
        digits = tok.replace(",", "")
        decimals = len(digits.split(".")[1]) if "." in digits else 0
        value = float(digits)
        if decimals == 0 and 1900 <= value <= 2100:
            continue  # a year
        before = [
            m
            for m in metrics
            if m[2] <= nm.start()
            and nm.start() - m[2] <= 30
            and not re.search(r";|\.\s", low[m[2] : nm.start()])
        ]
        after = [m for m in metrics if nm.end() <= m[1] <= nm.end() + 12]
        before = [m for m in before if _GAP.match(compare.strip_subjects(low[m[2] : nm.start()]))]
        hit = before[-1] if before else (after[0] if after else None)
        if hit is None:
            continue
        metric = hit[0]
        pct = low[nm.end() : nm.end() + 3].lstrip().startswith("%")
        i = nm.start()
        # An explicit minus must match a negative value; an unsigned figure may describe a fall in
        # words ("dropped 5.2%"), so it is matched by magnitude.
        minus = i >= 1 and low[i - 1] in _MINUS and (i < 2 or not low[i - 2].isalnum())
        if pct and metric.unit == "ratio":
            problems.append(f"{tok}% given for {metric.label}, which is not a percentage")
            continue
        cands = [lf for lf in all_leaves if lf.path[-1] in metric.keys]
        if not cands:
            problems.append(f"{metric.label} is not in the cited evidence")
            continue
        near = [
            s
            for s in subjects
            if hit[1] - 40 <= s[1] and s[2] <= hit[1] and ". " not in low[s[2] : hit[1]]
        ] or [s for s in subjects if nm.end() <= s[1] <= nm.end() + 30]
        if near:
            subject = near[-1][0] if near[-1][2] <= hit[1] else near[0][0]
            cands = [lf for lf in cands if compare.subject_matches(subject, lf.path)]
            if not cands:
                problems.append(f"{metric.label} for {subject} is not in the cited evidence")
                continue
        elif len(named) == 1:
            only = next(iter(named))
            cands = [lf for lf in cands if (lf.symbol or "").upper() == only] or cands
        if len(periods) == 1:
            want = next(iter(periods))
            known = [lf for lf in cands if lf.period is not None]
            if known and not any(lf.period == want for lf in known):
                problems.append(
                    f"{metric.label} cited is from the "
                    f"{'development' if want == 'test' else 'held-out'} period, not the "
                    f"{'held-out' if want == 'test' else 'development'} period described"
                )
                continue
            cands = [lf for lf in cands if lf.period in (want, None)]
        tol = 0.5 * 10**-decimals + 1e-9
        scales = (1.0, 100.0) if (pct or metric.unit == "pct") else (1.0,)
        target = -value if minus else value
        if not any(
            abs((lf.value if minus else abs(lf.value)) * k - target) <= tol
            for lf in cands
            for k in scales
        ):
            problems.append(f"{tok} does not match any {metric.label} value in the cited evidence")
    return problems


def unsupported_numbers(text: str, evidence: list[Evidence]) -> list[str]:
    """Numbers in `text` that appear in none of `evidence` (as value, rounded value or percent)."""
    blob = json.dumps([[e.args, e.data] for e in evidence], default=str)
    scrubbed = _EID.sub("", text)
    bad: list[str] = []
    for d in _DATE.findall(scrubbed):
        if d not in blob:
            bad.append(d)
    scrubbed = _DATE.sub("", scrubbed)
    pool = _numbers([[e.args, e.data] for e in evidence])
    for tok in _NUM.findall(scrubbed):
        clean = tok.replace(",", "")
        decimals = len(clean.split(".")[1]) if "." in clean else 0
        value = float(clean)
        if decimals == 0 and value <= 10:
            continue  # counts like "3 strategies"; digits this small cannot carry a fabrication
        if not _matches(value, decimals, pool):
            bad.append(tok)
    return bad


def text_problem(text: str, pool: list[Evidence], *, allow_synthetic: bool = False) -> str | None:
    """Checks shared by claims and uncertainty notes; `pool` is the evidence it may rest on."""
    if len(text) > MAX_CLAIM_CHARS:
        return "claim too long"
    if _LINKS.search(text):
        return "contains link or markup"
    if _ADVICE.search(text):
        return "contains trading advice or certainty language"
    if _EDGE.search(text):
        return "asserts predictive edge, contradicting held-out evidence"
    if _COMPARE.search(text):
        return (
            "free-text comparison cannot be verified; comparisons must be structured so code "
            "can evaluate them against the cited evidence"
        )
    if not allow_synthetic and _SYNTHETIC.search(text):
        return "refers to synthetic/demo data, but no synthetic data was used in this run"
    kinds = {m[0].cls for m in compare.metric_in(text.lower())}
    if "classification" in kinds and (kinds - {"classification"} or _RETURN_WORDS.search(text)):
        if _CONNECTOR.search(text):
            return "links classification performance to strategy returns without evidence"
    cites_backtest = any(e.tool == "get_backtest" for e in pool)
    if (cites_backtest or "strategy" in kinds) and _FORWARD.search(text):
        return "presents historical backtest results as an expectation of future performance"
    if cites_backtest and "strategy" in kinds and not _HISTORICAL.search(text):
        return "backtest figures must be labelled as historical/backtest results"
    problems = bound_metric_problems(text, pool)
    if problems:
        return "; ".join(problems[:3])
    bad = unsupported_numbers(text, pool)
    if bad:
        return f"numbers not found in cited evidence: {', '.join(bad[:4])}"
    return None


def check_claim(
    section: str,
    claim: Claim,
    by_id: dict[str, Evidence],
    *,
    allow_synthetic: bool = False,
) -> str | None:
    """Return a rejection reason, or None if the claim passes."""
    text = claim.text.strip()
    if not text:
        return "empty claim"
    unknown = [e for e in claim.evidence if e not in by_id]
    if unknown:
        return f"cites evidence not retrieved in this run: {', '.join(unknown[:3])}"
    cited = [by_id[e] for e in dict.fromkeys(claim.evidence)]
    if section == "verified_facts":
        if not cited:
            return "factual claim without evidence reference"
        if all(e.kind == "model" for e in cited):
            return "model output cannot be presented as a verified fact"
    elif section == "model_estimates":
        if not any(e.kind == "model" for e in cited):
            return "model estimate must cite model-generated evidence"
    pool = cited if section != "interpretation" or cited else list(by_id.values())
    return text_problem(text, pool, allow_synthetic=allow_synthetic)


def filter_claims(
    section: str,
    claims: list[Claim],
    by_id: dict[str, Evidence],
    *,
    allow_synthetic: bool = False,
) -> tuple[list[Claim], list[RejectedClaim]]:
    kept: list[Claim] = []
    rejected: list[RejectedClaim] = []
    for c in claims:
        reason = check_claim(section, c, by_id, allow_synthetic=allow_synthetic)
        if reason:
            rejected.append(RejectedClaim(section=section, text=c.text[:300], reason=reason))
        else:
            kept.append(Claim(text=c.text.strip(), evidence=list(dict.fromkeys(c.evidence))))
    return kept, rejected


def filter_notes(
    notes: list[str], by_id: dict[str, Evidence], *, allow_synthetic: bool = False
) -> tuple[list[str], list[RejectedClaim]]:
    """Uncertainty notes are free LLM text too: apply the evidence-wide text checks to them."""
    pool = list(by_id.values())
    kept: list[str] = []
    rejected: list[RejectedClaim] = []
    for n in notes:
        text = n.strip()
        reason = text_problem(text, pool, allow_synthetic=allow_synthetic) if text else "empty note"
        if reason:
            rejected.append(
                RejectedClaim(section="missing_or_uncertain", text=text[:300], reason=reason)
            )
        elif text:
            kept.append(text)
    return kept, rejected
