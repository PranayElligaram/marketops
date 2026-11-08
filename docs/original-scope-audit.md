# MarketOps original-scope gap audit (7 November 2026)

This document distinguishes the previously published **v1.0** release from the original-scope
extensions subsequently squash-merged to `main` in version **1.1.0**. It is not a claim of
production readiness. "Verified" means only the specific checks noted, not a universal guarantee.

| Capability | Agreed requirement | Existing v1.0 | New branch | Verification and remaining gap | Priority |
|---|---|---|---|---|---|
| Broker portfolio and history | Read-only positions, orders, cash and dividends | Implemented | Preserved | Historical account reconciliation previously tested; current release does not re-test the live broker | P1 |
| GBP valuations and SPY comparison | TWR, XIRR, drawdown, contribution-matched index | Implemented | Preserved | Synthetic checks pass in v1.0; new benchmark not yet tested on actual account | P1 |
| ML experiments | Logistic regression/XGBoost, purged walk-forward, SHAP, MLflow | Implemented | Preserved | Held-out AUC 0.496, no predictive advantage; survivorship bias remains | P1 |
| 2–12-week strategy evaluation | Deterministic entries/exits, next-open fills, costs, simple baseline | Not in v1.0; 5-session ML backtest exists | Fixed breakout/trend/close-based exit and matched buy-and-hold evaluator | Branch CI pending; does not establish forward profitability or point-in-time constituent coverage | P1 |
| Portfolio risk | Concentration, limited scenarios, insufficient-data states | Concentration implemented | Hypothetical GBP risk and concentration cap, prefilled from a previously loaded GBP portfolio snapshot where possible; technical screen | Branch CI pending; no new broker call; non-GBP or ambiguous instrument exposure requires manual inputs | P1 |
| Local research assistant | Ollama, tool-calling, provenance and verification | Implemented | Preserved | Local run previously tested; verifier is lexical and not proof of all factual claims | P1 |
| News retrieval | Free licensed, fresh sources where approved | No established current news pipeline | v1.2.0: SEC EDGAR filing metadata only | Filings: implemented, offline-fixture tested, live retrieval unverified (needs a contact User-Agent). Headlines: **blocked**, no free licensed fresh source ([details](research-sources.md)); never fabricated | P2 |
| Paper research journal | Record hypothetical decisions and subsequent outcomes | Not implemented | Private, manual PostgreSQL-backed journal with retrospective flag | v1.2.0: verified on real PostgreSQL; adds `closed_utc`, backfilled-outcome flag and prospective-only summary. Self-reported prices, not verified fills; **no forward outcomes exist yet** | P1 |
| Dashboard | Usable workflows, no misleading return claims | Implemented | Technical screen, strategy and journal UI | v1.2.0: browser-verified on synthetic data at 1280/1440 px; no defects found | P1 |
| Reproducibility and CI | Tests, docs, Docker, safe release | v1.0 completed with green GitHub Actions | Branch adds synthetic tests | Need full branch CI, PostgreSQL integration and smoke test before main/release | P1 |
| Real-account readiness | Read-only compatibility, honest reconciliation | Partial live checks in earlier development | No additional live calls | Real benchmark validation deliberately outstanding; private data never in CI | P1 |

## Research strategy boundary

The added fixed-rule research backtest is long-only, signal-at-close and hypothetical-fill-at-next-open,
with a 10 bp one-way cost assumption, minimum 10-session holding period for ordinary trend exits,
maximum 60 sessions, and a close-based breakdown exit which can occur sooner. Its execution is
a model of historical fills, **not** broker orders or a guaranteed stop. It compares the rule with
buy-and-hold from the same hypothetical first execution date to the same closing date.

This deterministic screen is **not** trained on the ML held-out test and must not be presented as
a validated profitable strategy. The research universe consists of current selected stocks and
has survivorship bias. Prices come from an unofficial personal-use provider and are not redistributed.

## Manual paper research record

Journal entries are deliberate user-created hypotheses with manually specified hypothetical
GBP entry, size, risk note and optional exit. They carry both the decision date and the actual
recording timestamp. Earlier decision dates are labelled retrospective. Gross illustrative P/L
is not execution-confirmed; spreads, FX, fees, taxes and slippage are not calculated. The journal
is private behind the local API token and lives in PostgreSQL, with no broker-write integration.

## Known limits not solved by new features

- No reliably measured predictive edge: 2022+ ROC-AUC ~0.496 against 0.5 chance.
- Point-in-time constituent membership and delisted-stock data remain unavailable.
- No automatically refreshing, provenance-checked current news retrieval.
- No proof of real-account SPY benchmark end-to-end correctness in the new release.
- No production-grade security operations, multi-user hosting or managed cloud service.
- No verified forward paper-trading performance history. Backfilled examples must remain labelled.
- No trading, order submission, account changes or auto-execution by design.

## Release verification (7 November 2026)

Merged into `main` via [PR #1](https://github.com/PranayElligaram/marketops/pull/1)
(squash commit `682ce198`). [Main CI run 42425408974](https://github.com/PranayElligaram/marketops/actions/runs/42425408974)
completed successfully: **654 passed, 9 skipped**; Ruff lint, Ruff formatting, strict Mypy,
pinned dependencies and API/dashboard Docker builds passed. The nine skipped tests are opt-in
PostgreSQL integration tests, not a passed PostgreSQL verification for this extension.

Still unverified in this extension: new journal persistence against a real PostgreSQL instance;
local Docker Compose startup and browser usability; authenticated live Trading 212 benchmark;
forward paper outcomes and reliable current news retrieval. These are release limitations, not
claims of functionality. The historical `v1.0.0` tag remains unchanged; package version is `1.1.0`.


## v1.2.0 scope status (8 November 2026)

- **Implemented and verified locally:** journal on real PostgreSQL (concurrency, restart persistence, migration),
  authentication, no broker surface, browser UX at 1280/1440 px, full pytest, Ruff, strict Mypy, Compose health.
- **Implemented but externally unverified:** live SEC EDGAR retrieval; real Trading 212 account and real-account
  benchmark (still outstanding; no read-only credentials were used); Ollama-generated answers.
- **Deferred by design:** forward paper outcomes (they need real elapsed time; none are fabricated), point-in-time
  index membership, hosted multi-user deployment, agent access to filings.
- **Blocked by provider/licence:** news headlines (see [research sources](research-sources.md)).
- The journal is single-user: there is no tenant or account model, only the one local access token.
