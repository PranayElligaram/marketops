# MarketOps

**AI-powered stock research and portfolio intelligence.** Local-first portfolio project targeting
AI engineering, ML, MLOps and Forward Deployed Engineering skills, with practical Trading 212 integration.

## Current state

Working end to end (details and exact numbers in [docs/status.md](docs/status.md)):

- **Data**: Yahoo chart endpoint (unofficial, personal use only, not redistributable) -> validated,
  idempotent CSV cache in `data/raw/` (git-ignored). 10 US large caps + SPY, daily OHLCV, 2005-present.
- **ML**: P(5-session forward return > 0). Point-in-time features, purged expanding-window walk-forward
  CV (2010-2021), single held-out test (2022+). Candidates: logistic regression, XGBoost.
- **Backtest**: signals at close, fills at next open, 10 bps one-way costs, vs momentum, equal-weight, SPY.
- **App**: FastAPI research endpoints + PostgreSQL persistence of predictions/model registry, Streamlit
  dashboard, local MLflow, SHAP, drift/data-quality monitoring.

### Honest headline result

**The model has no demonstrated predictive edge.** Held-out ROC-AUC is 0.496 (0.5 = chance), and it does
not beat an always-up baseline on log-loss or Brier. Walk-forward development AUC (0.514) did not hold
out of sample. XGBoost did not beat logistic regression, so logistic regression is retained. Backtests
of the model signal underperform simple equal-weighting after costs. The universe is current large caps,
so every strategy is inflated by survivorship bias. This is research software, not advice.

Trade execution endpoints do not exist in MarketOps. The Trading 212 integration is read-only.

### Research workflow

```
.\.venv\Scripts\python.exe -m marketops.data.cli ingest        # download + validate + cache
.\.venv\Scripts\python.exe -m marketops.data.cli status
.\.venv\Scripts\python.exe -m marketops.training.cli --mlflow  # train, evaluate, SHAP, drift report
.\.venv\Scripts\python.exe -m marketops.backtest.run           # writes backtest.json next to the model
docker compose up -d --build                                    # API mounts data/ and artifacts/ read-only
```

Endpoints (`http://127.0.0.1:8001/docs`): `/model/metadata`, `/model/evaluation`, `/model/explain`,
`/monitoring`, `/backtest`, `/market/{symbol}/history`, `/analysis/{symbol}`, `/analysis`,
`/predictions/{symbol}`, `/portfolio/example`, `/portfolio/overview`. MLflow UI:
`mlflow ui --backend-store-uri sqlite:///artifacts/mlflow.db`.

## Setup (Windows 11, PowerShell)

1. Extract to `C:\Users\prana\Workspace\Projects\MarketOps`, separate from existing FraudOps.
2. `Copy-Item .env.example .env`
3. Edit `.env` and replace `MARKETOPS_DB_PASSWORD` with a unique local password using letters, digits, underscores or hyphens (the current Compose URL does not escape other characters).
4. Run `docker compose up --build -d --wait`
5. Run `powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\verify_phase1.ps1`
6. Open `http://127.0.0.1:8503` or API docs at `http://127.0.0.1:8001/docs`.
7. Choose **Synthetic example** and click **Load portfolio** to verify the analytics.

### Enable your own Trading 212 account (optional)

Trading 212's API (beta) supports Invest and Stocks ISA accounts, not CFD or SIPP.
In Trading 212: Settings > API (Beta) > Generate API key. Select only portfolio/read permissions.
Use a **demo account key** with `MARKETOPS_T212_ENVIRONMENT=demo` first. When ready for your
real account, generate separate credentials and select `live` only after checking permissions.

Put keys only in your **local `.env`** (never in Git or a chat):

```
MARKETOPS_T212_ENVIRONMENT=demo
MARKETOPS_T212_API_KEY=...your local key...
MARKETOPS_T212_API_SECRET=...your local secret...
```

Restart with `docker compose up -d --force-recreate api` and reload the dashboard.
The UI fetches account data only when you press **Load portfolio**. No broker response is
persisted in PostgreSQL in this foundation. Keys are in the API container's environment and
must not appear in logs or screenshots. Do not publish or forward real portfolio screenshots.

### Tests (local Python)

```
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e '.[dev,train,ui,tracking]'
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\ruff.exe check .
.\.venv\Scripts\mypy.exe src api
```

### Ports

FraudOps already uses 8000 and 8502. MarketOps uses 8001 and 8503. Database ports are
not exposed to the host. Change `MARKETOPS_API_PORT` and `MARKETOPS_DASHBOARD_PORT` in `.env`
if needed; dashboard must be connected to internal API port 8000, not the published port.

## Scope and limits

- Position values are read from Trading 212's `walletImpact` fields in account currency;
  do not multiply local instrument prices across different currencies.
- Open-holdings unrealised P/L is not total account performance; excludes cash, fees,
  sales, closed positions and cash flows.
- No returns, risk estimates or forecasts exist until real historical data and model evaluation.
- Market data must be obtained under its provider licence. Do **not** publish downloaded price
  data or private portfolio data. Public GitHub release will contain source and reproducible instructions.
- Beta APIs can change. If the Trading 212 response schema changes, fail rather than
  silently calculate incorrect financial metrics.
- Cloud hosting and paid external APIs are not required.

## Roadmap

1. **Verified foundation:** Docker, read-only portfolio analytics and offline tests.
2. **Market dataset and ML pipeline:** EOD historical data with licensed local download;
   forecasting a 5-day *probability* of positive return, benchmark comparisons, rolling
   out-of-sample windows, leakage guards and realistic costs.
3. **Inference and versioning:** FastAPI, PostgreSQL, model versions, SHAP.
4. **Research agent:** Ollama reads only retrieved evidence, never sends trades.
5. **Dashboard:** analysis, historical backtests, concentration, monitoring.
6. **MLOps:** MLflow, tests, drift, CI, reproducible local Docker demo.
7. **Publish with approval**, after excluding credentials, brokerage data and licensed data.
8. **Full teaching guide**, based on the actual built system.

No unmeasured backtest performance will appear in this README.

## Data/API sources

- Official Trading 212 docs: https://docs.trading212.com/api
- Key creation: https://helpcentre.trading212.com/hc/en-us/articles/14584770928157-Trading-212-API-key
