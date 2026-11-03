# Research agent evaluation

Two separate things are evaluated. Do not mix them up.

| | What it measures | Needs Ollama | Deterministic |
|---|---|---|---|
| **Safeguard evaluation** (`tests/agent_eval`) | The verifier, comparison engine, tool allowlist, fallback and privacy handling, fed *scripted* model replies | No | Yes |
| **Live profile** (`scripts/profile_agent.py`) | Latency and token use of real local-model runs; the claims the real model produced and how many survived | Yes | No (CPU timing varies) |

Neither measures whether the market view is right. The ML model has no demonstrated edge; this tooling
checks that the agent's *statements* about its evidence are supported.

## Safeguard evaluation

```
python -m tests.agent_eval          # table of cases + metrics, exit code 1 on any failure
pytest tests/test_agent_eval.py     # the same set as a gate
```

26 cases on fixed artefacts (`tests/agent_eval/fixture.py`, constant metric values). Categories:

| Category | Cases | Examples |
|---|---|---|
| factual | 3 | last close and window return; held-out ROC-AUC; ML output stays a *model estimate* |
| supported comparison | 3 | model vs equal-weight CAGR; ROC-AUC vs chance 0.5; two tickers over an identical window |
| unsupported | 11 | wrong direction; different windows; two false prose comparisons taken from a real run; swapped metric; `%` on Sharpe; wrong strategy; held-out vs development period; backtest as forecast; AUC "explaining" returns; fabricated number / uncited / unknown id / ML-as-fact / advice; unsupported "synthetic" remark |
| missing data | 3 | unsupported symbol; no model artefacts; Ollama down |
| adversarial | 3 | injection in the question + `place_order`; path/URL/SQL smuggling in tool arguments; link markup in a claim |
| portfolio privacy | 4 | broker weights/tickers/keys absent from report and audit; broker refused when disabled; portfolio tool absent without selection; example data labelled synthetic |

Each case labels claims *valid* (must be accepted), *unsupported* (must be rejected, with a reason fragment)
and the expected tool-call set, rejected-call count, mode and status. Metrics:

- **claim acceptance** = labelled-valid claims accepted / labelled-valid claims
- **unsupported-claim rejection** = labelled-unsupported claims rejected with the expected reason and absent
  from every accepted section / labelled-unsupported claims
- **tool-call correctness** = cases whose executed (LLM-requested) tool set, forbidden-tool check and
  rejected-call count all match / cases

Results (2026-10-28, this repository state):

| | Original verifier | Hardened verifier |
|---|---|---|
| cases passed | 14 / 26 | 26 / 26 |
| claim acceptance | 15 / 18 (83.3 %) | 18 / 18 (100 %) |
| unsupported-claim rejection | 7 / 19 (36.8 %) | 19 / 19 (100 %) |
| unsupported claims accepted | 9 | 0 |
| tool-call correctness | 26 / 26 | 26 / 26 |

The original column was produced by running the same harness with the pre-change verifier logic
reconstructed from the earlier source. Two of the 19 unsupported items are *structured* comparisons the old
schema could not express, so they count as not rejected there; the 9 false accepts are all prose claims.
The set was written by the same engineer who wrote the verifier, so 100 % is a regression gate, not evidence
of coverage of unseen model behaviour. Extend it whenever a live run exposes a new failure.

### How comparisons are decided

The model does not write comparisons. It proposes `{metric, subject_a, evidence_a, subject_b, evidence_b,
relation}` (optionally `constant`). Code (`agents/compare.py`) looks both values up in the cited evidence,
requires the same tool, evaluation period/window and unit, computes the true relation (tolerance 5e-5), and
writes the sentence itself, with period dates and "point estimates; no significance test". Anything else is
rejected and shown with its reason. Free-text comparison wording (`higher`, `than`, `outperformed`, `vs` ...)
in a claim or an uncertainty note is rejected as unverifiable. This is a fail-closed guard; it also rejects
some harmless sentences, which is the intended trade-off.

### Other checks (all in `agents/verify.py`)

- every number next to a metric name must equal *that* metric's value for the named subject (model,
  equal-weight, SPY, always-up baseline, momentum, ticker), with units (`%` on Sharpe/AUC rejected) and sign
  (explicit minus must match a negative value);
- "held-out/test" prose may not use development numbers and vice versa;
- backtest figures must be labelled historical and may not be phrased as expectation or forecast;
- a sentence connecting a classification metric (AUC, Brier) to strategy returns is rejected;
- "synthetic/demo" remarks are rejected unless the synthetic example portfolio was selected;
- uncertainty notes (`missing_or_uncertain`) go through the same text checks as claims;
- claims beyond the per-section output limit are listed as rejected, never dropped silently.

## Live profile

```
python scripts/profile_agent.py out.json [--repeat N]
```

Runs three fixed questions against the configured local Ollama model and records per run: wall time, time in
LLM tool-selection vs synthesis calls, tool time, verification time, prompt/output tokens, tools called,
accepted/rejected claim counts and the rejected reasons. It wraps the HTTP layer from outside, so it can
measure any version of the agent. See `docs/status.md` for the recorded before/after numbers.

To assess a new model, run the profile and read the `claims` and `rejected_reasons` fields by hand; the
verifier decides *which* claims are shown, not whether the model's interpretation is useful.
