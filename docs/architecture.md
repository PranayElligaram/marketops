# MarketOps architecture

```mermaid
flowchart TD
    User[Investor] --> UI[Streamlit dashboard]
    UI -->|local HTTP, token for private routes| API[FastAPI]
    API --> Broker[Trading 212: fixed read-only GET calls]
    API --> History[Account history sync]
    History --> DB[(PostgreSQL)]
    DB --> Recon[Holdings + cash reconstruction, XIRR]
    DB --> Val[Daily NAV, TWR, drawdown]
    Prices[Local price / SPY / ECB FX cache] --> Val
    Val --> Bench[Contribution-matched SPY benchmark]
    API --> Demo[Synthetic demo account]
    Market[Local OHLCV CSV cache] --> Training[Walk-forward training + backtest]
    Training --> MLflow[MLflow tracking]
    Training --> Artifacts[Versioned, hash-checked model artifacts]
    Artifacts --> Inference[Inference + SHAP]
    API --> Inference
    API --> Agent[Local Ollama agent + evidence verifier]
    API --> Decision[Deterministic technical review + 2–12-week backtest]
    API --> Journal[Private manual hypothetical research journal]
    Journal --> DB
    Inference --> Agent
```

Layers: `src/marketops/` holds pure domain code (valuation, performance, ML, agent, broker adapter),
`api/` the FastAPI routes and the single authentication dependency, `app/` the Streamlit dashboard
(`perf_view.py` is formatting and charts only). Valuation, comparison and benchmark simulation are pure
functions over rows and cached files, so they are tested without network or database.

Design constraints: no code path for trades or transfers; read-only broker keys; private routes need a
local token and deny by default; no user position data in logs; fetched market data is cached
locally and never committed; no claim of ML performance without out-of-sample evaluation; a figure is
refused rather than estimated when its inputs do not reconcile.
