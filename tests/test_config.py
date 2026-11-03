"""Configuration and application-log tests."""

import json
import logging

from marketops.config import Settings
from marketops.logging_config import JsonFormatter


def test_settings_defaults_do_not_embed_a_database_password() -> None:
    settings = Settings(_env_file=None, database_url=None)
    assert settings.database_url is None
    assert settings.environment == "development"


def test_json_formatter_uses_structured_fields() -> None:
    formatter = JsonFormatter()
    record = logging.LogRecord("marketops", logging.INFO, __file__, 1, "alive", (), None)
    event = formatter.format(record)
    parsed = json.loads(event)
    assert parsed["level"] == "INFO"
    assert parsed["message"] == "alive"
    assert "timestamp" in parsed
