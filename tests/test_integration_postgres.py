"""Optional real PostgreSQL verification, enabled only when explicitly requested."""

import os

import pytest

from marketops.persistence.database import database_is_ready


@pytest.mark.skipif(
    os.getenv("MARKETOPS_RUN_INTEGRATION") != "1",
    reason="Set MARKETOPS_RUN_INTEGRATION=1 with MARKETOPS_DATABASE_URL to check real DB",
)
def test_postgres_is_ready() -> None:
    assert database_is_ready()
