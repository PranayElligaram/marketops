# Connecting Trading 212 (read-only, demo first)

MarketOps only issues two GET requests: `/equity/positions` and `/equity/account/summary`.
Trading 212's API supports Invest and Stocks ISA accounts only (not CFD), and returns values in the
primary account currency. Demo and live keys are separate.

1. In the Trading 212 app, switch to the **Practice (demo)** account, then Settings > API > generate a key.
   Grant only read scopes: **account**, **portfolio** and, for the Portfolio activity tab, the three history scopes (**history:orders**, **history:transactions**, **history:dividends**). Do not enable order/trading scopes. Without the history scopes the sync reports a scope error (HTTP 403) and the rest of the app keeps working.
   Copy the key ID and secret straight into `.env`; never paste them into a chat or issue.
2. In `.env` (git-ignored; keep it that way):
   ```
   MARKETOPS_T212_ENVIRONMENT=demo
   MARKETOPS_T212_ALLOW_LIVE=false
   MARKETOPS_T212_API_KEY=<key id>
   MARKETOPS_T212_API_SECRET=<secret>
   MARKETOPS_AGENT_API_TOKEN=<python -c "import secrets; print(secrets.token_urlsafe(32))">
   ```
3. `docker compose up -d` (recreates the api container with the new env).
4. Verify without printing secrets (PowerShell; the token is read from `.env` by you, not echoed):
   ```
   $t = Read-Host "Access token" -MaskInput
   Invoke-RestMethod http://127.0.0.1:8001/portfolio/overview -Headers @{ "X-MarketOps-Token" = $t } |
     Select-Object source, broker_environment, account_currency, holdings_value
   ```
   Expected: `source = trading212`, `broker_environment = demo`. Or use the dashboard Portfolio tab:
   choose "Trading 212 account", type the token, press Load portfolio.
5. Errors: 401 = wrong key/secret; 403 from Trading 212 = key missing a scope; 429 = wait a few seconds.
6. Live (real money) only after demo works: set `MARKETOPS_T212_ENVIRONMENT=live` **and**
   `MARKETOPS_T212_ALLOW_LIVE=true` with a live read-only key. The dashboard then shows a red LIVE banner.
   Revoke the key in the app when you stop using it.

## Account history (Portfolio activity tab)
Needs the three history scopes above and `MARKETOPS_AGENT_API_TOKEN`. Open **Portfolio activity**, enter the token, tick the
confirmation and press **Sync account history**. Nothing is downloaded on ordinary page loads. Trading 212 allows 20 history
requests per minute per endpoint (50 records per page), so a long history takes minutes; re-running is safe. See
`docs/historical-activity.md`.
