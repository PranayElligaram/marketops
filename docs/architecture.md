# MarketOps Architecture

```mermaid
flowchart TD
    User[Investor] --> UI[Streamlit Dashboard]
    UI -->|local HTTP| API[FastAPI]
    API --> Broker[Trading 212 read-only adapter]
    API --> Data[Portfolio Analytics]
    API --> DB[(PostgreSQL)]
    Historical[Locally downloaded EOD prices / CSV] -. next phase .-> Training[Walk-forward ML training]
    Training -. next phase .-> Tracking[MLflow]
    Tracking -. next phase .-> Inference[Versioned model inference]
    API -. next phase .-> Inference
    API -. next phase .-> Agent[Local Ollama research agent]
```

Design constraints: no trade execution capability; authenticated local keys with read-only permissions;
no user position data in logs; no fetched market data redistributed; no claims of ML performance without
real out-of-sample evaluation. PostgreSQL foundation is ready but the portfolio API currently remains
stateless and does not persist broker records.
