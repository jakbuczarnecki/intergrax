"""Real PostgreSQL runtime checks for transaction-local set_config semantics."""

from __future__ import annotations

import os

import pytest

from intergrax.integrations.providers.relational_store.postgresql.config import (
    PostgreSQLIntegrationConfig,
)
from intergrax.integrations.providers.relational_store.postgresql.session import (
    PostgreSQLConnectionProvider,
    set_local_config,
)
from platform_proofs.scenarios.verified_product_identification.qualification.integration.storage_environment import (
    postgres_environment_available,
)

pytestmark = [pytest.mark.integration]


def _provider() -> PostgreSQLConnectionProvider:
    integration = PostgreSQLIntegrationConfig.from_env(tenant_schema="public")
    return PostgreSQLConnectionProvider(integration, tenant_schema="public")


@pytest.mark.skipif(not postgres_environment_available(), reason="PostgreSQL not configured locally")
def test_set_local_config_application_name_visible_in_transaction() -> None:
    provider = _provider()
    marker = os.environ.get("INTERGRAX_POSTGRESQL_APPLICATION_NAME", "vpi-session-config-test")
    with provider.transaction() as session:
        set_local_config(session, "application_name", marker)
        row = session.execute(
            "SELECT current_setting('application_name') AS application_name"
        ).fetchone()
        assert row is not None
        assert row["application_name"] == marker


@pytest.mark.skipif(not postgres_environment_available(), reason="PostgreSQL not configured locally")
def test_set_local_config_statement_timeout_visible_in_transaction() -> None:
    provider = _provider()
    with provider.transaction() as session:
        set_local_config(session, "statement_timeout", "5000")
        row = session.execute(
            "SELECT current_setting('statement_timeout') AS statement_timeout"
        ).fetchone()
        assert row is not None
        assert row["statement_timeout"] in {"5000", "5000ms", "5s"}


@pytest.mark.skipif(not postgres_environment_available(), reason="PostgreSQL not configured locally")
def test_set_local_config_hostile_application_name_is_literal_value() -> None:
    provider = _provider()
    hostile = "'; DROP TABLE users; --"
    with provider.transaction() as session:
        set_local_config(session, "application_name", hostile)
        row = session.execute(
            "SELECT current_setting('application_name') AS application_name"
        ).fetchone()
        assert row is not None
        assert row["application_name"] == hostile


@pytest.mark.skipif(not postgres_environment_available(), reason="PostgreSQL not configured locally")
def test_set_local_config_does_not_persist_after_transaction() -> None:
    provider = _provider()
    marker = f"vpi-txn-local-{os.getpid()}"
    with provider.transaction() as session:
        set_local_config(session, "application_name", marker)
    with provider.connection() as session:
        row = session.execute(
            "SELECT current_setting('application_name', true) AS application_name"
        ).fetchone()
        assert row is not None
        assert row["application_name"] != marker
