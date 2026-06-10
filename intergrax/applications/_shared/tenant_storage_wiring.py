# © Artur Czarnecki. All rights reserved.

"""Postgres multi-tenant storage isolation wiring (AUDIT-IDEAL-4.2)."""

from __future__ import annotations

import re

from intergrax.applications.contracts.application_host import ApplicationProfile
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.integrations.providers.relational_store.postgresql.config import PostgreSQLIntegrationConfig
from intergrax.integrations.registry.profile import IntegrationProfile

_SCHEMA_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def tenant_schema_for_id(tenant_id: str) -> str:
    """Map tenant id to a safe PostgreSQL schema identifier."""
    normalized = re.sub(r"[^A-Za-z0-9_]", "_", tenant_id.strip().lower())
    if not normalized:
        normalized = "default"
    if normalized[0].isdigit():
        normalized = f"t_{normalized}"
    if not _SCHEMA_PATTERN.match(normalized):
        normalized = "tenant_default"
    return normalized[:63]


def resolve_tenant_postgresql_config(
    tenant_id: str,
    *,
    base: PostgreSQLIntegrationConfig | None = None,
) -> PostgreSQLIntegrationConfig:
    """Apply per-tenant ``search_path`` schema to a PostgreSQL integration config."""
    config = base or PostgreSQLIntegrationConfig.from_env()
    schema = tenant_schema_for_id(tenant_id)
    return config.model_copy(update={"tenant_schema": schema})


def integration_uses_postgresql(profile: IntegrationProfile) -> bool:
    binding = profile.relational_store
    if binding is None:
        return False
    return binding.resolved_slug() == "postgresql"


def product_requires_tenant_storage_isolation(env: ApplicationEnvironmentProfile) -> bool:
    """Product hosts with PostgreSQL relational store require tenant schema isolation."""
    if env.application_profile is not ApplicationProfile.PRODUCT:
        return False
    return integration_uses_postgresql(env.integration_profile)


def tenant_storage_isolation_ready(env: ApplicationEnvironmentProfile) -> bool:
    """Return True when product PostgreSQL posture declares tenant schema support."""
    if not product_requires_tenant_storage_isolation(env):
        return True
    options = env.integration_profile.options or {}
    raw = options.get("postgresql")
    if isinstance(raw, dict) and raw.get("tenant_schema"):
        return True
    env_schema = PostgreSQLIntegrationConfig.from_env().tenant_schema
    return bool(env_schema)
