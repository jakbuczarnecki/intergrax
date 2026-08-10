# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Databricks tenant-connection factory for restart-safe rehydration."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.providers.relational_store.databricks.bundle import (
    create_databricks_relational_store,
)
from intergrax.integrations.providers.relational_store.databricks.integration import (
    DATABRICKS_RELATIONAL_STORE_PROVIDER_ID,
    DatabricksRelationalStoreIntegration,
)
from intergrax.runtime.vendor_knowledge.models import JsonValue
from intergrax.runtime.vendor_knowledge.tenant_connection_rehydration import (
    TenantConnectionIntegrationFactory,
)

_ALLOWED_SECRET_FREE_CONFIG_KEYS = frozenset(
    {"host", "http_path", "catalog", "tenant_schema"}
)


def _require_nonblank(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def _parse_secret_free_config(
    value: Mapping[str, JsonValue],
) -> tuple[str, str, str | None, str | None]:
    if not isinstance(value, Mapping):
        raise ValueError("secret_free_config must be a mapping")
    unknown = set(value) - _ALLOWED_SECRET_FREE_CONFIG_KEYS
    if unknown:
        raise ValueError("secret_free_config contains unsupported keys")

    host = _require_nonblank(value.get("host"), field_name="host")
    http_path = _require_nonblank(value.get("http_path"), field_name="http_path")
    catalog = value.get("catalog")
    tenant_schema = value.get("tenant_schema")
    for field_name, field_value in (
        ("catalog", catalog),
        ("tenant_schema", tenant_schema),
    ):
        if field_value is not None and not isinstance(field_value, str):
            raise ValueError(f"{field_name} must be a string or null")
    return host, http_path, catalog, tenant_schema


class DatabricksTenantConnectionIntegrationFactory(TenantConnectionIntegrationFactory):
    """Compose one Databricks relational integration from durable connection data."""

    def __init__(
        self,
        *,
        connection_factory: Callable[[], Any] | None = None,
    ) -> None:
        self._connection_factory = connection_factory

    def create_integration(
        self,
        *,
        tenant_id: str,
        connection_ref: str,
        provider_id: str,
        integration_kind: IntegrationCategory,
        credential_ref: str,
        credential: str,
        secret_free_config: Mapping[str, JsonValue],
    ) -> DatabricksRelationalStoreIntegration:
        _require_nonblank(tenant_id, field_name="tenant_id")
        _require_nonblank(connection_ref, field_name="connection_ref")
        _require_nonblank(credential_ref, field_name="credential_ref")
        if provider_id != DATABRICKS_RELATIONAL_STORE_PROVIDER_ID:
            raise ValueError("provider_id does not match databricks")
        if integration_kind is not IntegrationCategory.RELATIONAL_STORE:
            raise ValueError("integration_kind does not match relational_store")
        token = _require_nonblank(credential, field_name="credential")
        host, http_path, catalog, tenant_schema = _parse_secret_free_config(
            secret_free_config
        )

        return create_databricks_relational_store(
            connection_factory=self._connection_factory,
            host=host,
            http_path=http_path,
            access_token=token,
            catalog=catalog,
            tenant_schema=tenant_schema,
        )


__all__ = ["DatabricksTenantConnectionIntegrationFactory"]
