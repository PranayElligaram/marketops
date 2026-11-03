# Technical audit: correctness and reproducibility (2026-10-28)

Scope: ML validity, backtest validity, reproducibility, API/DB/dashboard integration, CI. Previous-session claims were
treated as unverified. No Git repository exists, so nothing was committed. Nothing was tuned on the held-out test
period.

## 1. Verification of previous claims

| Claim | Result | Evidence |
|---|---|---|
| 56 passed / 1 skipped, ruff + strict mypy clean | **Reproduced** | Re-run before changes: 56 passed, 1 skipped; ruff and mypy clean |
| Held-out ROC-AUC 0.496 | **Reproduced** (0.4961, n=12,040) | Independent retrain into a scratch dir: predictions byte-identical, coefficients identical |
| Strategy CAGRs 14.3 / 16.1 / 17.3 % | **Reproduced**, with one change: model CAGR is now **14.4 %** after fix F2 moved the threshold 0.54666 to 0.54667 | `backtest.json`; independent equal-weight rebuild from raw CSV gave CAGR 17.34 % (stored 17.29 %), Sharpe 1.171 (1.168), max drawdown identical. The 0.05 pt gap is the start convention: the engine starts the equity curve one session earlier (at 1.0 before the first open) |
| Deterministic retraining | **Confirmed** (same data, same environment) | See above |
| Docker/local versions "not consistently pinned" | **Confirmed**; versions happened to match today but nothing enforced it | F3 |
| Docker services running | **Confirmed**; rebuilt, all three healthy | `docker compose ps` |
| "56 tests ... Postgres skipped" | Postgres test only checked `SELECT 1`; persistence was never tested | F6 |

## 2. Findings

Severity: High = could silently mislead results or provenance; Medium = real defect with limited blast radius;
Low = hygiene.

| ID | Sev | Finding | Evidence | Resolution |
|---|---|---|---|---|
| F1 | High | **Provenance record was wrong.** `write_manifest` overwrote the whole manifest with only the symbols of the latest ingest. `data/raw/manifest.json` and every model's `metadata["data"]` listed SPY only. Training had no content hash of its inputs. | Read of `manifest.json` and `metadata.json` | Manifest now merges; training records per-file SHA-256, rows and date range for all 11 CSVs (`metadata.data.fingerprint`); metadata also records config (horizon, seed, candidates, selection and threshold criteria). Existing manifest reconstructed from the cache and labelled as such. Test: `test_manifest_partial_reingest_keeps_other_symbols`, `test_training_is_deterministic_and_metadata_matches_artifact` |
| F2 | Medium | **Development labels reached into the test period.** 50 of 40,290 development rows (dates in the last 5 sessions of 2021) had `label_end` after 2021-12-31, so their labels used 2022 prices. Those rows fed fold-2021 OOF scores, model selection and the threshold. Training rows were purged correctly; evaluation rows were not. | `label_end > DEV_END` count = 50 | `dev_labelled()` excludes them. Selection unchanged (logreg C=0.01). Test AUC unchanged (0.49612). Threshold moved by 1e-5. Test: `test_dev_labels_never_reach_into_test_period` |
| F3 | Medium | **Dependencies unpinned.** pyproject ranges (`scikit-learn>=1.6,<2`) let Docker and CI resolve different versions from the venv; a pickled model loaded under another sklearn can silently misbehave. CI also did not install `mlflow`/`shap`, so those tests were skipped. | `pyproject.toml`, `Dockerfile`, `ci.yaml` | `constraints.txt` with exact versions, used by Docker (both stages), CI and local installs. CI installs the `tracking` extra (MLflow and SHAP tests now run), runs coverage, and fails on version drift. Inference refuses a model trained under a different scikit-learn version |
| F4 | Medium | **Inference trusted `LATEST` blindly.** No check that metadata describes the artefact, that the feature schema matches code, or that the pickle is the file that was trained. | `inference/service.py` | `load_latest` verifies version = directory, SHA-256 of `model.joblib`, feature list = `FEATURES`, `n_features_in_`, sklearn version, and a sane `LATEST` value. Models without a hash are refused (retrain). Test: parametrised `test_inference_rejects_mismatched_artifacts` |
| F5 | Medium | **Uncaught failures returned HTTP 500**: `/monitoring` with a missing CSV, `/predictions/{symbol}` with `predictions.csv` missing, `/model/evaluation` with a missing key. | Code path read; reproduced by regression test | Mapped to 503 or null fields. Test: `test_api_failure_states_are_503_not_500` |
| F6 | Medium | **Ingest could splice differently-adjusted history.** Yahoo rescales `adj_close` retroactively after each dividend. Ingesting a later partial window merged new rows onto old ones and left a fake return at the seam. Default full-history ingest is safe. | `ingest_symbol` merge logic | Refuses when overlapping `adj_close` differs (>1e-4) and cached rows outside the fresh window would be kept. Test: `test_ingest_refuses_splice_across_changed_adjustment` |
| F7 | Low | Backtest engine did not reject NaN or non-positive prices or NaN target weights (NaN would propagate silently into NAV). | Code read | Raises `ValueError`. Tests added for missing price, missing weight, non-trading signal day, multiple assets with a missing session, ex-dividend gap, CAGR, Sharpe and drawdown values |
| F8 | Low | `test_mlflow_logs_run` wrote placeholder artefacts into the repo's `./mlruns` (six 1-byte junk runs found). | `mlruns/1/*` | New experiments pin artefacts beside the tracking DB. Six junk run folders deleted (all had 1-2 byte placeholder files). Two real runs left. The existing experiment in `artifacts/mlflow.db` still points at `./mlruns`, so new runs on that DB continue to log there. Test: `test_mlflow_artifacts_do_not_pollute_cwd` |
| F9 | Low | Dashboard hardcoded "~55 % of windows were up" and "2022+ / 2010-2021" labels. | `app/dashboard.py` | Labels now derive from API metadata |
| F10 | Low | Postgres "integration test" was only `SELECT 1`. | `tests/test_integration_postgres.py` | Added round-trip test (registry idempotency, prediction insert and read back with JSON features) |

## 3. ML validity assessment (checked, no defect unless listed above)

- **Chronology.** Dev <= 2021-12-31 (12 expanding yearly folds from 2010), test >= 2022-01-01. Training rows for every
  fold and the final fit require `label_end < fold start` (purged). Dev evaluation rows are now also required to end by
  DEV_END (F2).
- **Feature leakage.** Features at t use only data <= t (`test_features_do_not_use_future_data` truncates future
  data and compares). Two caveats: (a) features use `adj_close`, which is rescaled by *future* dividends, so tiny
  look-ahead exists in absolute levels (the effect on returns is only the dividend factor; its size was not measured); (b) benchmark features are cut at each symbol's last date.
- **Preprocessing.** `StandardScaler` is inside the Pipeline and fit on training rows only (tested).
- **Selection and threshold.** Selected by OOF log-loss on development folds only; threshold by balanced accuracy on
  development OOF. The test period is evaluated once. `test_other_candidates_not_for_selection` is reported for
  transparency only. Note that the selection margin (logreg C=0.01 0.68556 vs C=1 0.68580) is within noise.
- **Baselines.** All models have worse OOF log-loss (0.6856) than an always-up prior (0.6846): the models are not
  better than predicting the base rate, even on development data. Held-out: Brier 0.24832 vs always-up 0.24769, log-loss
  0.68980 vs 0.68850.
- **Reported ROC-AUC.** 0.4961 on 12,040 rows is correct, but it overstates certainty. Rows are 10 stocks x
  overlapping 5-day labels x strongly cross-correlated, so the effective sample is far smaller. Date-block bootstrap
  (21-day blocks, 300 resamples; analysis script not retained) gives a 95 % interval of about [0.476, 0.518]. Mean daily
  cross-sectional rank IC = -0.006 (t about -0.25 after a 5-day overlap adjustment). Non-overlapping subsample AUC =
  0.490. Conclusion: indistinguishable from 0.5 and no evidence of edge.
- **Classification metrics.** F1 0.609 at threshold 0.547 is *below* the always-up F1 of 0.709 (the base rate is 54.9 %
  up), and balanced accuracy is 0.492. F1, precision and recall should not be read as skill. Calibration: mean predicted
  0.559 against 0.549 observed. By decile, observed up-rates are 0.54, 0.58, 0.57, 0.56, 0.53, 0.53, 0.53, 0.54, 0.53, 0.58 against predictions rising 0.51 to 0.61: non-monotone and over-confident in the upper deciles (predicted 0.57-0.59, observed 0.53), i.e. not calibrated out of sample.
- **Pooling.** AUC pools across dates, so it can reward predicting market-wide up periods; the cross-sectional IC above
  removes that effect and is also about zero.

## 4. Backtest validity assessment

Traced path: CSV -> `adj_close`, `open`, `close` -> features/labels -> probability at close t -> `p >= threshold` ->
equal weights across the selected names -> target on every 5th date -> engine executes at the **next session's
adjusted open** (`searchsorted(side="right")`) -> shares = weight x NAV after cost / adj open -> NAV marked at adj close
-> `performance()`.

- Timing: verified by test (signal-bar close price cannot be a fill price; changing future prices does not change
  earlier equity; Saturday signal fills next Monday).
- Dividends: `adj_close` is total-return adjusted and `adj_open = open x adj_close / close` is on the same basis,
  so dividends are included once. Tested with an ex-dividend gap. Cash earns 0 %.
- Costs: 10 bps on traded notional at every rebalance, including the first purchase. Same for model, momentum and
  equal-weight. The SPY benchmark follows the same engine and cost.
- Calendar: only dates present for all symbols are used. An asset missing a session removes that day for the whole
  portfolio.
- Metrics: CAGR on calendar years (days/365.25); volatility = daily std (ddof=1) x sqrt(252); Sharpe = mean daily
  return x 252 / vol, **risk-free rate 0** (so not an excess-return Sharpe); max drawdown from the equity curve
  including the initial 1.0. Values checked against hand computations in tests.
- Consistency: all strategies share rebalance dates, costs and the common calendar. Momentum is a fixed 126-day rule.
  Signals end 5 sessions before the data end (labels are required to form the prediction table), but positions are
  held to the last bar, so the equity curve runs to 2026-10-28. (The earlier status note was imprecise on this.)
- Not modelled: spread varying by stock or time, market impact, taxes, FX on Trading 212 (non-GBP stocks), fractional
  share limits, cash yield, intraday opens that differ from the auction price, and a short or leveraged book (the
  engine rejects them).
- Independent rebuild from the raw CSVs reproduced the stored equal-weight results (see section 1).

## 5. Reproducibility status

- Same data + same pinned environment -> identical predictions and coefficients (verified by a real retrain and by
  `test_training_is_deterministic_...`). The model *file hash* is not guaranteed identical across runs (timestamps in the
  version directory) and is not used for determinism.
- **Not reproducible from source:** the raw data. Yahoo's chart endpoint is unofficial, mutable, retroactively
  re-adjusted, has unclear licence terms, and cannot be re-downloaded to the same bytes later. The per-file SHA-256s
  recorded in each model's metadata identify exactly which CSVs were used, but those CSVs are not stored anywhere else
  and are not redistributable. Rebuilding from a future download will not reproduce today's numbers.
- Pinned: `constraints.txt` (23 packages). The Docker API image reports scikit-learn 1.9.1, xgboost 3.4.1,
  numpy 2.5.4 and pandas 2.3.3, matching metadata.
- Code version: no Git repository, so metadata cannot record a commit. Until Git is initialised the source cannot be tied
  to an artefact.

## 6. Integration results (executed)

- API in Docker against genuine artefacts: `/analysis/AAPL` -> probability 0.587, persisted to `prediction_log`
  (verified with `psql`), `/analysis/ZZZ` 404, `/analysis/SPY` 404, `days=2` 422, `/health/ready` ready.
- PostgreSQL integration tests run **inside the compose network** (5 passed, 0 skipped): readiness, DB layer, registry
  idempotency, prediction round trip.
- Streamlit (`AppTest`) against the live API: no exceptions; AUC and CAGR values in the rendered tables equal the API
  and `backtest.json`; API-down state shows a clear error.
- SHAP and MLflow ran for real in the retrain (`shap.json`, `monitoring.json`, run `9a8ccb0a...`) and via tests (no
  longer skipped in CI).
- Trading 212: still read-only (no order methods, fixed paths, no redirects, secrets from env only, errors do not echo
  credentials). **Live path remains untested** (no credentials requested or used).

## 7. Remaining risks and limitations

1. Survivorship bias: 10 stocks chosen *today* as large caps. Adding more current constituents would not remove this;
   only point-in-time membership including delisted names would, and the provider does not supply it. All backtest
   levels are inflated by an unknown amount.
2. No predictive edge was found. The conclusion is based on one 4.8-year test window that is dominated by a single
   regime and 10 correlated stocks; low power cuts both ways.
3. Equal-weight is the best strategy by construction of the universe (survivors); the comparison is not evidence about
   live trading.
4. Cost model is an assumption (10 bps flat); the model strategy turns over 34.6x a year and costs 22 % of initial NAV
   over the test window, so conclusions about it are cost-sensitive.
5. `adj_close` mild look-ahead and revisions (section 3). Data errors not detectable by validation (e.g. a bad but
   internally consistent bar) are not covered.
6. Pickle loading executes code. The SHA-256 check detects accidental mismatch, not a malicious attacker who can edit
   both files; artefacts are trusted-local only and mounted read-only in Docker.
7. No Git, no Alembic migrations (`create_all`), best-effort DB writes (an outage silently reports `persisted: false`),
   no authentication on the API (bound to 127.0.0.1 only).
8. `model_registry` registers a version only when `/model/metadata` is called.
9. CI still cannot run the real-data pipeline or the Postgres tests (no data or database in CI); the Postgres round
   trip is opt-in and was run manually in Docker. The old hash-less models (`...214140Z`, `...215542Z`, `...220008Z`)
   are now refused by inference if `LATEST` points at them.
10. The MLflow experiment in `artifacts/mlflow.db` predates the artefact-location fix, so new runs on it still write to
    `./mlruns`.
