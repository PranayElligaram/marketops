# MarketOps status: v1.2.0

Research and engineering project; **not investment advice and not production-ready**. The ML classifier has no
demonstrated predictive edge. Findings of the independent audit: [technical-audit.md](technical-audit.md).

## Verification of v1.2.0 (8 November 2026, local)
- Full `pytest`: 666 passed, 17 skipped (opt-in PostgreSQL tests).
- Real PostgreSQL 17 via Docker Compose (`scripts/run_integration_tests.sh`): 34 passed, 1 skipped (MLflow not in the
  slim image). New `tests/test_journal_postgres.py` covers create/read, NUMERIC(24,8) decimal and date persistence,
  retrospective flag, close and duplicate-close 409, four-way concurrent close (exactly one wins), persistence across
  connection restart, auth with nothing written, no broker/order surface, and `closed_utc` migration of a legacy table.
- `ruff check`, `ruff format --check`, `mypy --strict src api`: clean.
- Docker Compose: db, api, dashboard healthy; `/health/ready` ready; journal 401 without a token. A journal row created
  before `closed_utc` existed was migrated in place and reported as unknown, never guessed.
- Browser (Chrome; synthetic prices and portfolio on a throwaway Compose project with a throwaway token and no broker
  keys; 1280 and 1440 px css widths in same-origin frames): Portfolio, Portfolio activity, Historical performance with
  S&P 500 comparison, Stock research (technical screen, strategy report, risk scenario), AI agent unavailable/template
  fallback, Model performance, Backtests and Research journal (create, list, close, summary) render with no
  exceptions and no horizontal overflow. Journal create and close were confirmed in the database. No UI defects found.
- Fixed: a check-then-update race let two concurrent closes both pass the already-closed check; the update is now
  conditional on `status = 'open'`.
- Not verified: live SEC EDGAR retrieval; the real Trading 212 account and real-account benchmark; Ollama with a model
  present during the browser run; GPU. The CI result for the release commit is on GitHub Actions.

## Earlier verification (v1.1.0 source on main, 7 November 2026)
- [CI run 42425408974](https://github.com/PranayElligaram/marketops/actions/runs/42425408974):
  654 passed, 9 skipped; Ruff, format, strict Mypy, pinned versions and API/dashboard Docker builds passed.
- This release did not rerun optional PostgreSQL integration or a real browser smoke test for the new journal;
  prior v1.0 results below are historical, not verification of new features.

## Earlier v1.0 release verification
- `pytest`: 612 passed, 9 skipped (the skipped tests are the opt-in PostgreSQL integration tests, which run separately
  in Docker below).
- `ruff check`, `ruff format --check`, strict `mypy src api`: clean.
- PostgreSQL integration tests in Docker (`scripts/run_integration_tests.sh`): 26 passed, 1 skipped (MLflow is not in the
  slim image).
- Docker Compose: api, dashboard and db healthy; `/performance/demo` 200 without a token; private routes 401/403
  without one.
- Browser check (headless Chrome via Playwright, synthetic demo account): the Historical performance tab renders the
  benchmark section at 1280, 1440 and 1920 px with no horizontal overflow and no page errors.
- Not verified: any call to the real Trading 212 API from this release run (no credentials are used in CI or tests);
  Ollama inside CI; GPU performance.

## What is implemented
| Area | Code | Notes |
|---|---|---|
| Market data + ML research | `data/`, `features/`, `training/`, `backtest/`, `explainability/`, `monitoring/` | Walk-forward CV, held-out test, SHAP, MLflow, hash-checked artifacts |
| Research agent | `agents/`, `api/routes/agent.py` | Local Ollama, bounded tools, evidence verifier, deterministic fallback |
| Broker + history | `broker/t212.py`, `history/` | Read-only; idempotent PostgreSQL storage; completeness audit |
| Reconstruction + XIRR | `performance/` | Gated on reconciliation to the live snapshot |
| Valuation | `valuation/` | Daily GBP NAV, net invested, TWR, volatility, drawdown, period returns |
| Benchmark | `valuation/benchmark.py`, `benchmark_data.py`, `comparison.py` | Contribution-matched SPY total-return comparison ([design](benchmark-design.md)) |
| Demo | `marketops/demo.py`, `GET /performance/demo` | Deterministic synthetic account; no credentials |
| Dashboard | `app/dashboard.py`, `app/perf_view.py` | Streamlit + Altair; 1M/3M/6M/1Y/ALL windows |
| Security | `api/security.py`, `config.py` | Token on private routes (403 when unset), live double opt-in, redaction |

## Measured ML result (model 20261028T221802Z, logreg C=0.01)
- Walk-forward development (2010-2021, OOF): logreg AUC 0.514, XGBoost 0.503; always-up OOF log-loss 0.6846 beats every model.
- Held-out test (2022+, n = 12,040, base rate 54.9 %): **ROC-AUC 0.496** (date-block bootstrap 95 % about [0.476, 0.518]),
  non-overlapping AUC 0.490, PR-AUC 0.557, Brier 0.2483 (always-up 0.2477), log-loss 0.6898 (always-up 0.6885), F1 0.609
  (always-up 0.709), balanced accuracy 0.492. Not calibrated out of sample. **No predictive edge.**
- Backtest 2022+ (10 bps, 5-session rebalance, rf = 0): model CAGR 14.4 % / Sharpe 0.92; momentum_126d 16.1 % / 1.09;
  equal-weight 17.3 % / 1.17; SPY 11.9 % / 0.74. All survivorship-biased (10 current large caps).
- The development-period backtest was corrected in the audit (window bounded at `dev_end`); test-period numbers unchanged.

## Benchmark comparison (v1.0)
Contribution-matched SPY (dividend-adjusted close, converted to GBP with same-date ECB rates). Flows execute at the
close of their day or the next trading day; withdrawals never create negative holdings; no future prices are used.
Supported: portfolio and benchmark value, net invested capital, TWR (both), annualised TWR (>= 1 year), maximum
drawdown (>= 20 returns), closing-price XIRR (both), and the differences. Unavailable, with the exact reason shown:
no/invalid/stale benchmark data, FX gaps, benchmark gaps over 5 days, unvalued portfolio days inside the window,
withdrawals the benchmark cannot fund, contribution mismatch. `partial` when the portfolio is not reconciled.
The live-snapshot XIRR is not compared with a benchmark; the benchmark XIRRs end at a closing price (see the
design note). Not modelled: SPY expense ratio and tracking error against the index, costs, taxes, intraday timing.
The benchmark was verified on the synthetic account and with independent reference calculations; it was not run against
the author's real account in this release.

## Architecture decisions
- Pickled sklearn Pipelines (scaler inside, fit on train only); trusted local artefacts, hash/schema/version checked on load.
- Model selection by walk-forward OOF log-loss; threshold = balanced accuracy on development OOF.
- `constraints.txt` pins exact versions for local, CI and Docker.
- Database writes are best-effort so a DB outage never breaks read-only analysis; tables via `create_all` (no Alembic).
- No order or trade endpoints exist.

## Original-roadmap decision support

Version 1.1.0 adds deterministic 2–12-week historical research, a technical screen, a
hypothetical GBP risk-budget scenario and a private manual research journal. These are distinct
from the five-session ML model and do not demonstrate a trading edge. PostgreSQL and
browser validation of the journal are recorded above (v1.2.0). v1.2.0 adds SEC filing metadata
([research sources](research-sources.md)) and prospective-versus-backfilled journal tracking.
See [the original-scope audit](original-scope-audit.md).

## Known limitations
- Survivorship bias in the ML universe; no point-in-time membership data is available from the provider.
- Raw price data is not reproducible from source (unofficial Yahoo endpoint, retroactive adjustment, personal-use
  licence). Only SHA-256 fingerprints of the CSVs used are recorded; no second price source.
- Reconstruction cannot prove history reaches account opening; broker corporate actions make an instrument unvalued;
  ticker renames need a local override.
- The stored `dev_oof` equity curve of the oldest model versions overlapped the test period (older `backtest.json` files
  are superseded); current artefacts record explicit period bounds.
- The agent verifier is lexical/rule-based: it binds numbers to metrics, subjects and periods but cannot judge arbitrary
  semantics. A 4B local model on CPU takes about a minute per answer.
- Trading 212 beta API: history endpoints need `history:*` read scopes; semantics of dividend `amount` (net vs gross)
  are undocumented.
- Local single-user design; no hosted deployment.
