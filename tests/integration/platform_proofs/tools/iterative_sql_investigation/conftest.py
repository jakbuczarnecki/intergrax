# © Artur Czarnecki. All rights reserved.

"""Docker-backed fixtures for TOOLS-ITERATIVE-SQL-INVESTIGATION."""

from __future__ import annotations

import os
from collections.abc import Generator

import pytest

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.relational_store.postgresql import create_postgresql_relational_store
from platform_proofs.tools.iterative_sql_investigation.dataset import bulk_load_parcel_events
from platform_proofs.tools.iterative_sql_investigation.runtime import (
    ADMIN_DSN_ENV,
    DEFAULT_ADMIN_DSN,
    DEFAULT_RUNTIME_DSN,
    DSN_ENV,
    ProofSqlRuntime,
    build_proof_sql_runtime,
)

# Start:
#   docker compose -f platform_proofs/tools/iterative_sql_investigation/docker-compose.yml up -d
#   uv run pytest tests/integration/platform_proofs/tools/iterative_sql_investigation -m "integration and network" -q


def _resolve_admin_dsn() -> str | None:
    return os.environ.get(ADMIN_DSN_ENV, DEFAULT_ADMIN_DSN).strip() or None


def _resolve_runtime_dsn() -> str | None:
    return os.environ.get(DSN_ENV, DEFAULT_RUNTIME_DSN).strip() or None


def _open_admin_store():
    dsn = _resolve_admin_dsn()
    if dsn is None:
        pytest.skip(f"Requires {ADMIN_DSN_ENV} or default local proof admin DSN")
    try:
        store = create_postgresql_relational_store(dsn=dsn, tenant_schema="proof")
        store.connect()
        return store
    except (IntegrationConfigurationError, ConnectionError, TimeoutError, OSError) as exc:
        pytest.skip(f"Proof PostgreSQL admin backend unavailable: {type(exc).__name__}: {exc}")


@pytest.fixture
def proof_sql_dataset_loaded() -> Generator[None, None, None]:
    store = _open_admin_store()
    try:
        bulk_load_parcel_events(store, row_count=300)
    finally:
        store.close()
    yield


@pytest.fixture
def proof_sql_runtime(proof_sql_dataset_loaded: None) -> Generator[ProofSqlRuntime, None, None]:
    dsn = _resolve_runtime_dsn()
    if dsn is None:
        pytest.skip(f"Requires {DSN_ENV} or default local proof runtime DSN")
    try:
        runtime = build_proof_sql_runtime(dsn=dsn)
    except (IntegrationConfigurationError, ConnectionError, TimeoutError, OSError) as exc:
        pytest.skip(f"Proof PostgreSQL runtime backend unavailable: {type(exc).__name__}: {exc}")
    try:
        yield runtime
    finally:
        runtime.close()
