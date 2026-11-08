"""Runtime settings; credentials belong only in environment variables."""

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MARKETOPS_", env_file=".env", extra="ignore")

    environment: Literal["development", "test", "production"] = "development"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    database_url: str | None = None
    t212_environment: Literal["demo", "live"] = "demo"
    # Live (real-money account) reads need this explicit opt-in on top of t212_environment=live.
    t212_allow_live: bool = False
    t212_api_key: str = ""
    t212_api_secret: str = ""
    data_dir: str = "data/raw"
    model_dir: str = "artifacts/models"
    # Writable local cache of provider prices and ECB FX used for historical valuation.
    market_cache_dir: str = "data/market"
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "qwen3.5:4b"
    agent_timeout_s: float = 300.0
    agent_max_iterations: int = 6
    agent_max_tool_calls: int = 8
    # Broker-backed portfolio data may only reach the local LLM when explicitly enabled.
    agent_allow_broker_portfolio: bool = False
    # Required (header X-MarketOps-Token) for any agent request that touches broker data or
    # reads the agent audit log while broker access is enabled. Empty = broker path refused.
    agent_api_token: str = ""
    # SEC fair access requires a descriptive User-Agent with a real contact, e.g.
    # "Jane Doe jane@example.com". Empty disables the filings endpoint; nothing is invented.
    sec_user_agent: str = ""

    @property
    def broker_configured(self) -> bool:
        return bool(self.t212_api_key and self.t212_api_secret)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
