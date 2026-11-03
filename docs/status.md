# MarketOps Status (updated 2026-11-03, after Historical Performance dashboard milestone)

Full findings: `docs/technical-audit.md`. Not production-ready: this is a research tool with no demonstrated
predictive edge.

## Verified in the audit
- Tests: 74 passed, 2 skipped (both opt-in Postgres tests; run separately in Docker: 5 passed, 0 skipped).
  Ruff, `ruff format --check` and strict mypy (`src api`) clean. Coverage 79 % (training/data/backtest-run CLIs are 0 %).
- Held-out ROC-AUC 0.496 reproduced; retraining on identical data and environment gives identical predictions.
- Independent rebuild of the equal-weight backtest from raw CSVs matches stored results (CAGR 17.34 % vs 17.29 %).
- Docker API/dashboard/db healthy; prediction persistence confirmed in PostgreSQL; Streamlit tables equal API data.

## Defects fixed (see audit F1-F10)
Provenance manifest overwritten and no input hashes; development labels reaching into the test period (50 rows);
unpinned dependencies and CI skipping MLflow/SHAP; inference without hash/schema/version checks; HTTP 500s on missing
artefacts; ingest splicing re-adjusted prices; backtest accepting NaN; tests polluting `./mlruns`; hardcoded dashboard
text; trivial Postgres test.

## Current model 20261028T221802Z (logreg C=0.01, threshold 0.5467)
- Walk-forward dev (2010-2021, OOF): logreg AUC 0.514, XGB 0.503; always-up OOF log-loss 0.6846 beats every model (0.6856+).
- Held-out test (n=12,040, base rate 54.9 %): AUC 0.496 (date-block bootstrap 95 % about [0.476, 0.518]),
  non-overlapping AUC 0.490, PR-AUC 0.557, Brier 0.2483 (always-up 0.2477), log-loss 0.6898 (always-up 0.6885),
  F1 0.609 (always-up 0.709), balanced accuracy 0.492. Not calibrated out of sample. **No predictive edge.**
- Backtest 2022+ (10 bps, 5-session rebalance, rf = 0): model CAGR 14.4 % / Sharpe 0.92 / turnover 34.6x;
  momentum_126d 16.1 % / 1.09; equal-weight 17.3 % / 1.17; SPY 11.9 % / 0.74.
  *Discrepancy with the earlier report:* model CAGR was 14.3 % before audit fix F2 (threshold changed by 1e-5).
  All figures are inflated by survivorship bias and are valid only under the documented assumptions.

## Architecture decisions
- Pickled sklearn Pipelines (scaler inside, fit on train only); trusted local artefacts, hash/schema/version checked on load.
- Selection by walk-forward OOF log-loss; threshold = balanced accuracy on development OOF.
- `constraints.txt` pins exact versions for local, CI and Docker.
- DB writes are best-effort so a DB outage never breaks read-only analysis. Tables via `create_all` (no Alembic).
- No order or trade endpoints exist; Trading 212 access is read-only.

## Known limitations / blockers
- Survivorship bias (10 current large caps). More current constituents would not fix it; point-in-time membership
  including delisted names is needed and is not available from the provider.
- Raw data is not reproducible from source (unofficial Yahoo endpoint, retroactive adjustment, unclear licence). Only
  the SHA-256 fingerprints of the CSVs used are recorded.
- No Git repository, so artefacts cannot reference a code commit.
- Cost model is a flat assumption; Sharpe uses rf = 0; cash earns 0 %.
- CI cannot run real-data training or Postgres tests. Old hash-less model versions are refused by inference.
- Existing MLflow experiment still logs artefacts to `./mlruns` (new DBs use `mlartifacts/`).
- Trading 212 live path untested (no credentials used).

## Research agent (milestone 2026-10-28)
Code: `src/marketops/agents/` (`tools`, `runner`, `verify`, `ollama`, `schemas`), `api/routes/agent.py`,
"AI research agent" tab in `app/dashboard.py`, `agent_run` audit table, tests `tests/test_agent*.py`.
- **API:** `POST /agent/research {question, portfolio_source: none|example|trading212}`, `GET /agent/status`,
  `GET /agent/runs` (audit). Local Ollama only (URL host allowlist: loopback, `host.docker.internal`, `ollama`).
- **Tools (allowlist, pydantic `extra=forbid`):** list_supported_symbols, get_price_summary, get_ml_signal,
  get_model_metadata, get_model_evaluation, get_backtest, get_shap_importance, get_portfolio_exposure
  (only offered when the request selects a portfolio; broker source additionally needs
  `MARKETOPS_AGENT_ALLOW_BROKER_PORTFOLIO=true` plus read-only keys; weights only, no values, withheld from
  responses and audit). No shell/filesystem/network/SQL/order tools exist.
- **Bounds:** 6 iterations, 8 tool calls, 300 B arg limit, 6 kB result limit, 1000-char question, 300 s total.
- **Loop:** tool-calling phase, then a schema-constrained synthesis call. Each claim must cite evidence ids
  (E1..); verifier rejects uncited/unknown citations, numbers absent from cited evidence, ML output presented as
  fact, advice/certainty language, claims of predictive edge, links/markup. Rejected claims are shown, not hidden.
  The no-edge / calibration disclosure is generated deterministically from the retrieved evaluation, not by the LLM.
- **Fallback:** Ollama down, model missing, timeout, malformed output, no evidence or all claims rejected give a
  template summary labelled `deterministic_fallback` with the reason; no model-written analysis is invented.
- **Model:** `qwen3.5:4b` (tools capable). Compared with `qwen2.5:7b-instruct` on the same AAPL question (CPU only):
  both completed the loop (144 s vs 124 s); qwen3.5:4b produced more verified claims and fewer rejections.
- **Verified:** ruff, ruff format, strict mypy clean; pytest 115 passed, 2 skipped (opt-in Postgres); no test needs
  Ollama or credentials (CI unchanged). Real Ollama runs: local (AAPL, 144 s) and via the Docker API container to
  host Ollama (demo portfolio + MSFT/TSLA + injection attempt, 95 s; TSLA reported as unsupported, injection
  flagged, run persisted to PostgreSQL). Docker api/dashboard/db healthy.
- **Unverified:** broker-backed portfolio path against the real Trading 212 API (no credentials; only a stub);
  Ollama inside CI; GPU performance.
- **Limits:** verifier checks numbers, citations and wording, not the semantic truth of comparisons or of free
  text in `missing_or_uncertain` (the LLM wrote one wrong remark about a "synthetic universe" in a live run). A 4B
  model on CPU takes 1.5-2.5 minutes. The agent covers only the 10-stock trained universe and has no fundamentals or news.

## Agent hardening milestone (2026-10-29)
Details of the evaluation: `docs/agent-evaluation.md`. Code: `agents/compare.py` (new), `agents/verify.py`,
`agents/runner.py`, `api/routes/agent.py`; tests `tests/test_agent_verify.py`, `test_agent_hardening.py`,
`test_agent_eval.py`, `tests/agent_eval/`; profiler `scripts/profile_agent.py`.

**Defects found and corrected**
- Verifier accepted numerically false prose in a real qwen3.5:4b run: "ROC-AUC 0.4961 ... lower than the
  momentum baseline (0.4944)" (false) and "slightly lower max drawdown (-16.03%) vs equal-weight (-17.39%)"
  (ambiguous). Cause: only number presence was checked. Now comparisons must be structured and are evaluated
  by code (same tool, period/window, unit; true direction computed); prose comparison wording is rejected.
- Numbers were not bound to metric/subject/period/unit/sign (a Sharpe value could be reported as ROC-AUC,
  `92.4%` as a Sharpe, development numbers as held-out). Now bound and checked.
- Backtests could be phrased as future expectations; AUC could be "explaining" returns; uncertainty notes
  (`missing_or_uncertain`) were unverified. All now checked. "Synthetic" remarks allowed only for the example portfolio.
- Real-run bug: the model re-requesting the pre-fetched evaluation (E1) counted as "no evidence" and replaced a
  valid answer with a template. Fixed. Over-limit claims are listed as rejected, never dropped silently.
- Broker-backed research and (when broker access is enabled) the audit listing had no access control. They now
  need `MARKETOPS_AGENT_API_TOKEN` (16+ chars, header `X-MarketOps-Token`, constant-time compare), else 403/401;
  sensitive evidence hashes are withheld from reports and audit rows.
- Backtest evidence now carries `period_start/period_end`. **Pre-existing artefact inconsistency found:** the
  stored `dev_oof` equity curve runs 2010-01-04 to 2026-10-28 although `dev_end` is 2021-12-31, so it overlaps the
  held-out period. Not changed (out of scope, no threshold/result changes); the tool now emits `period_warning`.
  The `dev_oof` backtest figures should not be quoted as development-only until this is investigated.

**Safeguard evaluation** (26 deterministic cases, scripted model replies; regression gate, not an LLM score):
original verifier 14/26 cases, 9 of 19 labelled-unsupported claims accepted, acceptance 15/18, rejection 7/19;
hardened 26/26, 0 false accepts, acceptance 18/18, rejection 19/19, tool-call correctness 26/26.

**Latency, real local Ollama (qwen3.5:4b, CPU), same 3 questions, one run each** (`scripts/profile_agent.py`):

| question | before | after |
|---|---|---|
| price + signal (AAPL) | 121.2 s | 57.5 s |
| backtest vs baselines | 123.4 s | 58.4 s |
| held-out edge | 24.5 s (fell back to template: bug above, not equivalent) | 58.2 s (full LLM answer) |

Before: 99.4 % of time in model calls (tool turns 29-39 s, synthesis 82-94 s, 1,100-1,300 output tokens); tools
0.02-0.7 s; verification under 5 ms. Changes: one tool round (the extra "done?" model turn is skipped when every
call succeeded), E1 not re-offered, compact tool results during selection, bounded synthesis (item/length caps
in the JSON schema, `num_predict` limits), no calibration bins unless asked. Output tokens fell to 550-690.
Costs of the speed-up: at most 4 facts / 2 estimates / 2 interpretations / 3 comparisons / 3 notes per answer.
Single runs on a shared CPU: expect about +/-10 % variation. Verification, bounds and quality checks unchanged.

**Verification run on 2026-10-29:** pytest 187 passed, 2 skipped (opt-in Postgres); ruff, ruff format, strict
mypy (`src api`) clean; Streamlit AppTest and FastAPI tests included in the suite; real Ollama runs (3 local
profile runs plus one through the rebuilt Docker API container, 61.9 s, persisted to PostgreSQL); Docker
api/dashboard/db healthy; broker request without token refused (403).

**Remaining limitations**
- qwen3.5:4b never produced a structured comparison in live runs, so comparisons are rejected, not answered;
  answers state facts side by side. Prompt/model tuning is open. Prose guard is fail-closed and rejects some
  harmless sentences (about 1-5 per live answer), all shown as rejected.
- Verifier is lexical/rule-based: it binds numbers to metrics/subjects/periods but cannot judge arbitrary
  semantics (e.g. an unsupported causal story with no numbers). Unsigned figures match by magnitude.
- Evaluation set is small and written by the verifier's author; live-model quality is only spot-checked.
- Other API routes (`/portfolio/overview`) remain unauthenticated and rely on the loopback-only port binding;
  the agent token covers only agent endpoints. Compose does not yet pass `MARKETOPS_AGENT_API_TOKEN`.
- Still CPU-bound at about 1 minute per answer; no Trading 212 live path tested.

**Read-only Trading 212 readiness:** not yet. Needed first: authenticate or remove `/portfolio/overview`, wire the
token through Compose/dashboard, and test the real broker response shape with a demo-environment key (needs user
action). The agent-side controls (opt-in flag, token, weights only, withheld evidence/hashes, no write tools) are in place.

## Verified historical valuation and time-weighted performance (2026-11-03)
Design, conventions and limits: `docs/valuation.md`. Code: `src/marketops/valuation/`, routes
`GET /performance/valuation` and `POST /performance/valuation/refresh` (token protected), dashboard section
under the Historical performance tab. No model, threshold, agent or broker-write change; XIRR is untouched.

**Implemented:** instrument mapping verified against the account's own fills (fully sold instruments included,
GBX/GBp handling, local override file); daily bars cache with null-bar removal; ECB reference FX through
Frankfurter with freshness/completeness validation; event-driven daily NAV (shares from executed fills, provider
split restatement, ambiguity refusal when the broker also reports a corporate action, bounded price/FX staleness,
negative-cash guard); cash ledger identical to the broker-reconciled ledger (dividends only as cash, never via
adjusted prices); net invested capital vs NAV; time-weighted return with explicit flow timing and a start-of-day
sensitivity; segments across gaps; annualised volatility (>= 60 returns), maximum drawdown (>= 20), annualised
return (>= 1 year); monthly/annual returns with a `complete` flag; reconciliation of split-adjusted shares and
last-close NAV to the live snapshot; Streamlit charts (NAV vs net invested, TWR index, drawdown, period tables).

**Live read-only verification (private figures deliberately not recorded):**
- All instruments ever traded, including a fully sold one, mapped to provider symbols and verified from their own
  fills (100 % of fills inside the daily range; currencies agree). One renamed legacy ticker initially FAILED
  verification (the provider assigns the old symbol to a different company) and was fixed with a local override,
  which then verified; this is the intended behaviour.
- Price coverage: every weekday from the first record to the snapshot date valued (no unvalued days) after null
  holiday bars were dropped (two were found and initially produced unvalued days). Some days use a prior close and
  are flagged stale. FX: USD and EUR legs available for the whole period; the account currency is the base.
- Reconciliation: share counts equal the broker's; last-close NAV agrees with the live total to well under the 1.5 %
  `match` tolerance; the overall valuation status is `verified`.
- Metrics available: NAV, net invested capital, net gain, time-weighted return (+ timing sensitivity), annualised
  TWR, annualised volatility, maximum drawdown, monthly and annual tables.
- Sanity: NAV positive on every day, no daily time-weighted return beyond 15 %, TWR and NAV gain have the same sign.
- Defects found live: the ECB bulk CSV served stale data (now rejected; Frankfurter used); a ticker rename; null
  provider bars; the zero placeholder in the ECB CSV for one currency.

**Tests:** 533 passed, 9 skipped (opt-in PostgreSQL) locally; Ruff, `ruff format --check` and strict mypy
(`src api`) clean; Docker PostgreSQL integration 26 passed, 1 skipped (mlflow absent from the slim image) including
a valuation test on PostgreSQL; api/dashboard/db healthy; `/performance/valuation` 401 without the token. New tests
(`test_valuation_core.py`, `test_valuation_report.py`): holdings and cash, splits, multiple currencies, missing
FX/prices, flow timing, deposits/withdrawals, dividends, TWR, drawdown, reconciliation failures, API
authentication and dashboard rendering. The dashboard was exercised with Streamlit AppTest, not in a browser.

**Limits:** free personal-use price data with no second source for closes; no interpretation of mergers,
spin-offs or ticker changes (a renamed symbol must be overridden by hand); corporate-action fills from the broker
make that instrument unvalued from that date; the account had no corporate action, so real split handling is
verified only on synthetic data; daily closes only; benchmark comparison not implemented.

## Exact next action
Initialise Git (with approval) so artefacts can record a commit, then decide the next milestone: a point-in-time,
survivorship-aware dataset and a cost/regime sensitivity analysis. Agent follow-up when wanted: prompt/model tuning so structured comparisons are used; investigate the `dev_oof` curve overlap.

## Access-control and dev-period milestone (2026-10-29)

### Part A: private portfolio access
- **Root cause:** `GET /portfolio/overview` had no authentication; the dashboard read the token from its own
  environment (a proxy risk if it was ever set there); compose never passed `MARKETOPS_AGENT_API_TOKEN` to the API,
  so the existing agent guard could not be enabled in Docker; the audit stored the first 200 chars of broker-backed
  questions.
- **Boundary:** public = `/portfolio/example`, agent `portfolio_source` none/example (synthetic, no credentials).
  Private = `/portfolio/overview`, `/agent/research` with `trading212`, `/agent/runs` (when broker agent access is on).
  All private routes use one dependency (`api/security.py`): token in `X-MarketOps-Token`, `hmac.compare_digest`,
  minimum 16 chars, **403 when no token is configured** (deny by default), 401 when missing/wrong. Auth runs before any
  broker call, so unauthenticated callers learn nothing about broker configuration.
- **Dashboard:** no token in its environment or compose service. The user types it into a password field (held in the
  Streamlit session only) and it is sent only on the real-portfolio / real-agent requests; synthetic requests send none.
- **Audit/logging:** broker-backed `question_preview` is `withheld` (hash kept); evidence data/hashes were already
  withheld; broker errors are fixed strings; tests assert key, secret and token never appear in responses or logs.
- **Config:** `MARKETOPS_AGENT_API_TOKEN` added to `compose.yaml` (api only) and `.env.example`.
- **Not verified:** the real Trading 212 API (no credentials; fake client/mock transport only).

### Part B: development backtest period
- **Root cause:** `backtest()` marked NAV on every calendar date to the end of the price history. The dev signals
  were clean (last signal 2021-12-23) but the final dev position was held and valued through 2026-10-28, so reported
  dev metrics included 2022-2026 returns. This was a **reporting/evaluation-window error in the backtester, not
  model contamination.**
- **Verified clean:** dev OOF predictions are walk-forward, purged (max `label_end` 2021-12-31), 2010-01-04 to
  2021-12-23; no date overlap with test (2022-01-03 onward); threshold and model choice use dev OOF only.
  The held-out test period was never used for any decision here and is untouched; test curves are bit-identical
  before/after the fix.
- **Fix:** `backtest(..., end=)` cuts the calendar (signals that would execute after `end` are dropped, not shifted);
  `run_split` rejects predictions outside the split bounds; `run_backtests` rejects `dev_end >= test_start` and
  overlapping dates, and records `periods` (bounds, first/last signal, first/last equity date) in `backtest.json`.
- **Regenerated:** `artifacts/models/20261028T221802Z/backtest.json` (older model versions' `backtest.json` still have
  the unbounded dev curves and are superseded). Deterministic: re-run gives identical curves (tested).
- **Changed dev_oof metrics (2010-01 to 2021-12), CAGR / Sharpe:** model 16.4 % / 1.01 -> 17.0 % / 1.02;
  momentum_126d 17.2 % / 1.07 -> 18.4 % / 1.10; equal-weight 17.6 % / 1.13 -> 18.2 % / 1.12; SPY 14.1 % / 0.86 ->
  14.9 % / 0.91. Max drawdowns unchanged. **Test (2022+) metrics unchanged.** Dev numbers remain survivorship-biased.
- Tests: `tests/test_backtest_periods.py` (boundary, later-price independence, dropped late signals, overlap and
  metadata mismatch rejection, reproducibility); mutation check confirmed it fails if the boundary is removed.

## Trading 212 read-only analytics milestone (2026-10-29)
Code: `broker/t212.py`, `portfolio/analytics.py`, `api/routes/portfolio.py`, "Portfolio" tab in `app/dashboard.py`,
`config.py`, `compose.yaml`, `scripts/run_integration_tests.sh`; tests `tests/test_t212_contract.py`;
user guide `docs/trading212-setup.md`.

**Audit against docs.trading212.com (checked 2026-10-29):** HTTP Basic key:secret (matches the adapter);
`https://demo|live.trading212.com/api/v0`; positions limit 1 req/s, account summary 1 req/5 s; Invest and Stocks ISA
only; values in the primary account currency; API is beta. Positions fields used: `instrument.{ticker,name,isin,currency}`,
`quantity`, `walletImpact.{currency,currentValue,totalCost,unrealizedProfitLoss,fxImpact(nullable)}`. Summary fields:
`currency`, `totalValue`, `cash.{availableToTrade,inPies,reservedForOrders}`,
`investments.{currentValue,totalCost,realizedProfitLoss,unrealizedProfitLoss}`. Fixtures are built from this schema, not
from a real response (the docs give no full example).
**Gap found:** no account summary (so no cash), rate-limit/scope/timeout errors not distinguished, no live opt-in.

**Changes:** client now has exactly two fixed-path GETs (`get_positions`, `get_account_summary`), no write surface, no
redirects, safe error strings for 401/403/408/429/5xx/network; demo default, live needs `T212_ENVIRONMENT=live` **and**
`T212_ALLOW_LIVE=true` (also enforced for the agent). Overview adds cash, total incl. cash, unrealised P/L %, concentration
(top-1, top-3, HHI, effective N, warnings), exposure by instrument currency and ISIN prefix, FX impact, data-quality
warnings, `broker_environment`. 10 s in-API snapshot cache protects the broker rate limit (errors never cached). Cash is
optional: a failed or inconsistent summary removes cash only. Dashboard: source/environment banner (SYNTHETIC / DEMO / LIVE),
account currency and snapshot time, metrics, concentration warnings, allocation + currency charts, holdings table with
P/L, empty and error states. Historical performance is explicitly "not available" (no transaction/cash-flow ingestion).

**Postgres integration import problem:** the DB port is not published, and tests run in the image could not import `api`
(no pyproject/pythonpath). `scripts/run_integration_tests.sh` mounts the repo read-only in a Compose one-off container.

**Verified:** pytest 242 passed, 2 skipped (opt-in Postgres, run separately); Postgres tests in Docker 19 passed, 1 skipped
(MLflow not in the slim image); ruff, ruff format, strict mypy (`src api`) clean; images rebuilt, api/dashboard/db healthy;
ports bound to 127.0.0.1; `/portfolio/overview` without token 403 (no token configured); `/portfolio/example` 200.
**Not verified:** any real Trading 212 call. No credentials are configured (`broker_configured=False`), so demo
connectivity, real response shapes (e.g. whether `totalValue` includes cash, `fxImpact` semantics) are unconfirmed. No live requests made.

**Limitations:** no historical performance, dividends, fees, realised P/L per position; no sector/country exposure
(ISIN prefix is issuer registration, not economic exposure); no FX rates are fetched or invented; single account; the
agent still receives weights only; concentration thresholds (25 % / 60 % / 5 holdings / 50 % foreign) are rule-of-thumb
guides; stray empty directories `pyproject.toml;C`, `pyproject.toml;W`, `tests;C`, `tests;W` in the repo root predate this
milestone and were left untouched.

## Trading 212 final offline review (2026-10-29)
**Checked, no change needed:** documented token variable is `MARKETOPS_AGENT_API_TOKEN` everywhere (docs, compose, dashboard,
API); no `T212_TOPS` variant exists. Client remains two fixed GETs; demo default and live double opt-in unchanged. Position
P/L, cost, value come from `walletImpact` (account currency); currency exposure uses `instrument.currency` as a label only.
Auth/scope/rate-limit/5xx/network errors map to safe messages; malformed summary values drop cash only; malformed position
values return 502 with no totals.
**Defects fixed:** (1) account with zero positions ignored the summary, so cash was lost and currency was "N/A"; the summary
currency is now used. (2) Combined total was `positions + cash` without evidence that `totalValue`/cash fields are additive;
the docs describe `totalValue` only as "Investments value" and do not say whether `reservedForOrders`/`inPies` overlap
`availableToTrade`. The combined total and cash % are now shown only if the broker's `totalValue` equals
`investments.currentValue` + the three cash fields; otherwise they are hidden with a data-quality warning (distinct message
when `totalValue` equals investments only). (3) Dashboard would have crashed formatting a missing combined total; it now
shows "Unverified".
**Tests added (4, `tests/test_t212_contract.py`):** empty positions + cash, totalValue excluding cash, unreconciled
totalValue, malformed cash value.
**Verified offline:** pytest 246 passed, 2 skipped (opt-in Postgres); ruff check, ruff format --check, strict mypy `src api`
clean; api/dashboard rebuilt and healthy; `/portfolio/example` 200 (GBP, total 1250 reconciled, cash 20.0%);
`/portfolio/overview` without token 403; dashboard health 200. Docker Postgres integration: 19 passed, 1 skipped (mlflow absent from slim image).
**Not verified:** any authenticated Trading 212 request. Real `totalValue` semantics remain unconfirmed until the first demo
response; if the combined total shows "Unverified", that response is the evidence needed to decide.

## Trading 212 historical activity foundation (2026-10-29)
Details and data contracts: `docs/historical-activity.md`. Code: `broker/t212.py` (history GETs, `nextPagePath` validation,
pacer), `history/{parse,sync,read}.py`, `persistence/models.py` (4 new tables), `api/routes/history.py`, "Portfolio activity" tab
in `app/dashboard.py`, reconciliation in `portfolio/analytics.py`; tests `test_history_{parse,client,sync,api,dashboard,postgres}.py`,
`test_portfolio_reconciliation.py`, `tests/conftest.py`.

**Part A (live analytics audit).** Re-checked against the OpenAPI schemas. Added a broker-vs-computed reconciliation (value, cost
basis, unrealised P/L: `match` / `drift` / `mismatch` / `not_reported`); a missing `realizedProfitLoss`/`totalCost`/
`unrealizedProfitLoss` no longer removes cash. **Live check (read-only, no figures recorded):** summary and positions
agree on currency, cash parsed, combined total verified (so `totalValue` = investments + cash on the real account), cost and
value reconcile; unrealised P/L differed by 0.009 % of holdings value, which is price movement between the two separate
requests (hence the new `drift` status, tolerance 0.1 % of value). **Test-isolation defect fixed:** tests read the real `.env`
(`test_broker_research_refused_without_configured_token` failed once the user configured a token); `tests/conftest.py` now
strips `MARKETOPS_*` and the `.env` file for every test.

**Parts B-D.** Read-only retrieval of orders, cash transactions and dividends; idempotent PostgreSQL storage keyed by
`(environment, account_key, source_id)`; per-page commits; incremental only after a finished full pass; interrupted/failed/stale/
partial states; conflict, duplicate, rejected and inconsistent counters; private `/history/*` routes behind the existing token;
`POST /history/sync` is the only broker-touching route and the dashboard needs an explicit confirmation tick. Create-if-missing
tables follow the project convention (no Alembic).

**Part E.** `GET /history/readiness` reports which metrics are verified. Computed now: net deposits/withdrawals (gated),
dividend income, cash fees and interest, reported realised P/L (only if it reconciles to the broker's all-time figure), counts
and per-currency order values. **Unavailable by design:** XIRR, time-weighted return, equity curve, total return %, benchmark.

**Verified:** pytest 417 passed, 7 skipped (opt-in Postgres); ruff, ruff format, strict mypy (`src api`) clean; Docker
PostgreSQL integration 24 passed, 1 skipped (MLflow absent from the slim image) including 5 history tests on a throwaway database
(unique constraints, Decimal/timezone round trip, idempotency, interruption recovery); api/dashboard/db rebuilt and healthy;
`/history/status` and `/portfolio/overview` 401 without token, `/portfolio/example` 200.

**Not verified (live):** all three history endpoints returned HTTP 403 (scope) for the configured key, so real history response
shapes, record counts and the live classification of `TRANSFER`/dividend types are unconfirmed; parsers follow the OpenAPI schema
and synthetic fixtures only. Fix: enable the read-only scopes `history:orders`, `history:transactions`, `history:dividends` on
the Trading 212 API key. No full sync was run against the live account (needs your deliberate action); the probe fetched at
most the first page per endpoint and stored nothing.

**Known gaps:** deposit/withdrawal amount sign convention and whether dividend `amount` is net or gross are undocumented (types
define direction; amounts are magnitudes/as credited); `TRANSFER` is not counted; the API cannot prove history reaches account
opening; incremental sync misses back-dated records (forced full pass fixes); history is not re-validated against positions;
no valuation history exists, so no returns; dashboard shows sync progress only on manual refresh.

## Re-verification of the historical activity foundation (2026-11-03)
The milestone above was already implemented; nothing was rebuilt. Re-run: pytest 417 passed, 7 skipped (opt-in Postgres); ruff,
ruff format and strict mypy (`src api`) clean. Live read-only probe (first page per endpoint, counts only, nothing stored):
orders, transactions and dividends all still return HTTP 403 (key lacks `history:*` scopes), so live history remains unverified.
Docker Desktop was not running, so the Docker/PostgreSQL integration and health checks were not re-run (last passed 2026-10-29).

## Verified historical portfolio reconstruction and performance (2026-11-03)
Code: `src/marketops/performance/{xirr,holdings,report}.py`, `api/routes/performance.py` (`GET /performance/historical`, token
protected), "Historical performance" tab in `app/dashboard.py`; tests `test_perf_{xirr,holdings,report}.py` (synthetic only) and one
PostgreSQL test in `test_history_postgres.py`. No model, threshold, agent or broker-write change; the agent still receives no history.

**Method.** Fills are replayed chronologically (time, order id, fill id) up to the broker snapshot time `as_of`; later records are
ignored (tested). Average-cost basis; partial fills summed; oversold positions stay negative and are flagged; corporate-action and
unclassified fills are flagged and **not applied**; inconsistent or foreign-currency fills are excluded and flagged. Positions are
compared with the live snapshot by ticker (tolerance 1e-6 shares); cost is graded match/drift/mismatch (0.5 % / 2 %, informational).
A **cash ledger** (deposits - withdrawals - fees + interest + all dividend-type credits - purchases + sales) is compared with the live
cash balance (tolerance 1.00 + 0.01 % of purchases); this also serves as the inception test (zero opening balance reproduces today's
cash and positions: consistent with, not proof of, full history). XIRR uses investor-signed deposits/withdrawals plus the verified
terminal value; a grid search finds all roots in [-99.9 %, +100,000 %]; ambiguous (several roots), no-solution, <2 flows, one-sided
flows, <30 days, naive dates and non-finite values are refused, and a sub-year period carries a warning.

**Verified data coverage (live account; figures deliberately omitted from the public repository).** All three datasets reached
the last page with consistent page arithmetic, stored count equalled imported count, identifiers were unique, nothing was rejected,
conflicting or inconsistent, only the account currency occurred, and the sign convention was uniform.

**Reconciliation (live read-only snapshot).** All positions match in quantity and cost; cash ledger matches the broker balance to
well inside tolerance; per-fill realised P/L sums to the broker's all-time figure and the average-cost reconstruction agrees with it;
the XIRR equals an independent closed-form check (single root, no ambiguity). Overall status
`verified`.

**Metrics implemented (each gated; otherwise `partial` with blockers or `unavailable` with a reason):** external deposits,
withdrawals, net invested capital (+ cumulative net-invested chart); dividends received; cash interest and dividend-interest; account
fees; realised P/L (broker-reported, cross-checked); unrealised P/L snapshot; net gain on invested capital (verified value + withdrawals
- deposits); money-weighted return (XIRR).

**Unavailable and why.** Time-weighted return, equity curve, total return %, benchmark comparison: no historical valuations exist (only
the current snapshot), no historical prices for the instruments held, no FX history and no corporate-action records are stored; a curve
would be invented. XIRR is also unavailable whenever any gate fails (incomplete sync, TRANSFER/unsupported types, foreign-currency
flows, position or ledger mismatch, no live snapshot, unverified broker total).

**Defects found while testing:** a deposit in a non-account currency was summed into the account-currency total (fixed: excluded and
flagged); an unfilled order created an empty reconstructed position (fixed); the page-count audit assumed full pages (now `>=`).

**Verified (2026-11-03):** pytest 467 passed, 8 skipped (opt-in Postgres); ruff, ruff format --check and strict mypy (`src api`) clean;
Docker PostgreSQL integration 25 passed, 1 skipped (mlflow absent from the slim image) including the new report test; api/dashboard/db
rebuilt and healthy; `/performance/historical` 401 without token; live report 200 with `verified`. The dashboard tab is covered by
Streamlit AppTest against the real FastAPI app; it was not opened in a browser.

**Remaining discrepancies / limits.** None found on the live account. Not proven: that history starts at account opening (only
consistent with it); whether dividend `amount` is net or gross; ISIN is not stored on fills, so mapping is by ticker only (a ticker
change would surface as a mismatch, not be repaired); no corporate action occurred in the data, so split handling is flag-only; the
XIRR is dominated by two deposits at the start (it is effectively an annualised total return) and is sensitive to deposit timing;
the report needs a fresh sync to reproduce cash exactly (events after the last sync show up as a ledger mismatch, not silently).

## Exact next action
Add the three history scopes to the key, press **Sync account history** (Portfolio activity tab), then review the completeness
table, the `TRANSFER`/unclassified counts and the reported-realised-P/L reconciliation. Holdings reconstruction and XIRR are now done (section above). Next milestone: benchmark comparison (same dates and currency), a second price source to cross-check closes,
and interpretation of broker-reported corporate actions.

## Historical Performance dashboard redesign (2026-11-03)
Presentation-only change; calculations, API, ML and agent code untouched. See "Dashboard presentation" in
`docs/valuation.md`. New `app/perf_view.py`; tab rewritten in `app/dashboard.py`; tests in `tests/test_perf_view.py`
(synthetic fixtures only), wording assertions updated in `tests/test_perf_report.py` and `tests/test_valuation_report.py`.
- Verification: unit tests of KPI values/states/period filtering/chart specs against API-shaped data, and Streamlit
  AppTest renders (these are not a browser). Browser check: one manual Chrome pass against a synthetic stub API
  (structure, KPI grid, charts, tables, expanders rendered). The browser viewport was extremely large and screenshot
  zoom failed ("Detached"), so fine visual density at common desktop widths was NOT visually confirmed in a browser.
- UI wording changed from "verified" to reconciliation-status labels; API statuses are unchanged.

