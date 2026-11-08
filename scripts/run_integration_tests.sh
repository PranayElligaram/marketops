#!/usr/bin/env sh
# Runs the opt-in PostgreSQL tests inside the Compose network (the DB port is not published).
# The repository is mounted read-only at /work so pyproject's pythonpath (src, .) resolves
# `api` and `marketops`. The DB URL comes from the api service definition; no secret is read here.
set -eu
cd "$(dirname "$0")/.."
MSYS_NO_PATHCONV=1 docker compose run --rm --no-deps \
  -e MARKETOPS_RUN_INTEGRATION=1 -e HOME=/tmp -e PIP_NO_WARN_SCRIPT_LOCATION=1 \
  -v "$PWD:/work:ro" -w /work api sh -c \
  "python -m pip install -q --user -c constraints.txt pytest && \
   python -m pytest -p no:cacheprovider -q tests/test_integration_postgres.py tests/test_audit_regressions.py tests/test_history_postgres.py tests/test_journal_postgres.py"
