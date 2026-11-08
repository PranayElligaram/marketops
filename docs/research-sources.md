# Company and news research sources

Assessed 8 November 2026 against the original requirement of fresh, licensed company/news research
at £0. Only what is stated below was checked; terms can change.

## Integrated: SEC EDGAR filing metadata (company disclosures, not news)

| Question | Finding |
|---|---|
| Licence / permitted use | SEC says sec.gov information is public and may be copied or redistributed without permission, and asks for appropriate citation. SEC names and logos are trademarks and are not used. |
| Attribution | Every response carries "Source: U.S. Securities and Exchange Commission, EDGAR (www.sec.gov)." |
| Freshness | The submissions API is documented as updated in real time as filings are disseminated (typically under a second). MarketOps caches results for 15 minutes in process. |
| Cost / key | Free, no key or authentication. |
| Rate limit | 10 requests per second per user (SEC fair access). The client spaces requests at 0.25 s, caches, and makes at most two calls per lookup. |
| Identification | Automated clients must declare a descriptive User-Agent with a real contact. MarketOps requires `MARKETOPS_SEC_USER_AGENT`; if unset the endpoint returns 503 and makes no network call. No contact is invented. |
| Provenance | `GET /research/filings/{symbol}` returns CIK, accession number, form, filing date, acceptance time (UTC), item numbers, a link to the primary document on www.sec.gov, the source URL and `retrieved_utc` (when SEC was actually queried; a response served from the 15-minute cache keeps its original fetch time). |
| Validation | The ticker must exist in SEC's own ticker map. Rows must match the accession and document-name patterns, be an allowed form (8-K, 10-K, 10-Q, 6-K, 20-F, 40-F, DEF 14A) and not be dated in the future; bad rows are dropped and structural errors fail closed. Document links are built by MarketOps, never copied from the response. |
| LLM involvement | None. The data is shown by the dashboard (Stock research tab) and API only; the agent has no filings tool, so it cannot invent or paraphrase filings. |

Limits: US-registered issuers only. Filing metadata says nothing about a company's prospects and
contains no headlines or analysis.
**Live retrieval from sec.gov was not exercised in this environment** (no contact address was
configured for the User-Agent); behaviour is verified against offline fixtures that follow the documented
response shape.

## Blocked: general news headlines

No source was found that is simultaneously free, licensed for storing and summarising headlines in this
application, fresh and reliable:

- Publisher and wire services (Reuters, Bloomberg, FT and similar) and aggregator APIs are paid or
  restrict redistribution and derived use.
- Yahoo Finance headlines/RSS are not licensed for application use (the project already treats the Yahoo
  price endpoint as personal-use and unofficial) and are not relied on.
- Free-tier news APIs impose delay, caps or non-commercial/attribution conditions that were not
  verified, so none is integrated.

Until a source with an explicit licence and a verified freshness contract is chosen, MarketOps has no news
pipeline and the agent must not state any recent headline or market event. No site is scraped.
