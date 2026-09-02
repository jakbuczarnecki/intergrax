# © Artur Czarnecki. All rights reserved.

"""IntegrationProfile builders — vendor setup isolated to fixtures."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from intergrax.integrations.contracts.base import UnknownIntegrationError
from intergrax.integrations.providers.relational_store.postgresql.config import (
    PostgreSQLIntegrationConfig,
)
from intergrax.integrations.registry.catalog import get_entry
from intergrax.integrations.providers.relational_store.postgresql.register import (
    register_postgresql_integration,
)
from intergrax.integrations.providers.relational_store.sqlite.register import (
    register_sqlite_integration,
)
from intergrax.integrations.registry.catalog_manifests import POSTGRESQL, SQLITE
from intergrax.integrations.registry.profile import IntegrationProfile

POSTGRESQL_DSN_ENV = "INTERGRAX_COLLABORATIVE_WORK_POSTGRESQL_DSN"
POSTGRESQL_SCHEMA_PREFIX = "collab_e2e_"


def ensure_relational_integrations_registered() -> None:
    try:
        get_entry(SQLITE.slug)
    except UnknownIntegrationError:
        register_sqlite_integration()
    try:
        get_entry(POSTGRESQL.slug)
    except UnknownIntegrationError:
        register_postgresql_integration()


def sqlite_integration_profile(data_dir: Path) -> IntegrationProfile:
    ensure_relational_integrations_registered()
    return IntegrationProfile(
        relational_store=SQLITE,
        options={SQLITE.slug: {"data_dir": str(data_dir)}},
    )


def postgresql_integration_profile(
    *,
    dsn: str,
    schema_name: str,
    connection_factory: object | None = None,
) -> IntegrationProfile:
    ensure_relational_integrations_registered()
    options: dict[str, Any] = {
        "dsn": dsn,
        "schema_name": schema_name,
    }
    if connection_factory is not None:
        options["connection_factory"] = connection_factory
    return IntegrationProfile(
        relational_store=POSTGRESQL,
        options={POSTGRESQL.slug: options},
    )


def resolve_postgresql_dsn() -> str | None:
    explicit = os.environ.get(POSTGRESQL_DSN_ENV, "").strip()
    if explicit:
        return explicit
    base = PostgreSQLIntegrationConfig.from_env()
    return base.connection_string() or None


def invalid_postgresql_profile() -> IntegrationProfile:
    ensure_relational_integrations_registered()
    return IntegrationProfile(
        relational_store=POSTGRESQL,
        options={
            POSTGRESQL.slug: {
                "dsn": "postgresql://invalid:invalid@127.0.0.1:1/unreachable",
            }
        },
    )
