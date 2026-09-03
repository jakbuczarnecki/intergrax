# © Artur Czarnecki. All rights reserved.

"""PostgreSQL fixtures for Autonomous Work integration tests."""

from __future__ import annotations

import os
import uuid
from collections.abc import Generator

import pytest

from intergrax.autonomous_work.persistence import (
    AutonomousWorkRepositories,
    open_postgresql_autonomous_work_repositories,
)
from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.relational_store.postgresql.config import (
    PostgreSQLIntegrationConfig,
)

DSN_ENV = "INTERGRAX_AUTONOMOUS_WORK_POSTGRESQL_DSN"
SCHEMA_PREFIX = "autonomous_work_test_"

# Run (PostgreSQL must be reachable — e.g. infra/docker/postgresql/docker-compose.yml):
#   docker compose -f infra/docker/postgresql/docker-compose.yml up -d
#   set INTERGRAX_AUTONOMOUS_WORK_POSTGRESQL_DSN=postgresql://intergrax:intergrax@localhost:5434/intergrax
#   uv run pytest tests/integration/autonomous_work/test_postgresql_repository.py -m "integration and network" -q


def resolve_postgresql_config() -> PostgreSQLIntegrationConfig | None:
    dsn = os.environ.get(DSN_ENV, "").strip()
    if not dsn:
        base = PostgreSQLIntegrationConfig.from_env()
        if not base.connection_string():
            return None
        return base
    return PostgreSQLIntegrationConfig(dsn=dsn)


def open_bundle(schema_name: str) -> AutonomousWorkRepositories:
    config = resolve_postgresql_config()
    if config is None:
        pytest.skip(
            f"PostgreSQL Autonomous Work tests require {DSN_ENV} "
            "or INTERGRAX_POSTGRESQL_* connection settings"
        )
    try:
        return open_postgresql_autonomous_work_repositories(
            config=config,
            schema_name=schema_name,
        )
    except (IntegrationConfigurationError, ConnectionError, TimeoutError, OSError) as exc:
        pytest.skip(f"PostgreSQL backend unavailable: {type(exc).__name__}: {exc}")


def drop_schema(schema_name: str) -> None:
    config = resolve_postgresql_config()
    if config is None:
        return
    bundle = open_postgresql_autonomous_work_repositories(
        config=config,
        schema_name="public",
    )
    try:
        with bundle.store.transaction() as conn:
            conn.execute(f"DROP SCHEMA IF EXISTS {schema_name} CASCADE")
    finally:
        bundle.close()


@pytest.fixture
def postgresql_autonomous_work_bundle() -> Generator[AutonomousWorkRepositories, None, None]:
    schema_name = f"{SCHEMA_PREFIX}{uuid.uuid4().hex}"
    bundle = open_bundle(schema_name)
    try:
        yield bundle
    finally:
        bundle.close()
        drop_schema(schema_name)
