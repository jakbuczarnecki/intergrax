# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.providers.project_status.client import HttpxProjectStatusReadClient
from intergrax.integrations.providers.project_status.config import ProjectStatusIntegrationConfig
from intergrax.integrations.providers.project_status.integration import ProjectStatusIntegration
from intergrax.integrations.providers.project_status.knowledge_read import (
    PROJECT_STATUS_PROVIDER_ID,
)
from intergrax.runtime.vendor_knowledge.tenant_connection_factory_contract import (
    EagerTenantConnectionIntegrationFactoryMixin,
)
from intergrax.runtime.vendor_knowledge.models import JsonValue

_ALLOWED_SECRET_FREE_CONFIG_KEYS = frozenset({"base_url", "timeout_seconds"})


def _require_nonblank(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def _parse_secret_free_config(
    value: Mapping[str, JsonValue],
) -> ProjectStatusIntegrationConfig:
    if not isinstance(value, Mapping):
        raise ValueError("secret_free_config must be a mapping")
    unknown = set(value) - _ALLOWED_SECRET_FREE_CONFIG_KEYS
    if unknown:
        raise ValueError("secret_free_config contains unsupported keys")
    base_url = _require_nonblank(value.get("base_url"), field_name="base_url")
    timeout = value.get("timeout_seconds")
    if timeout is not None and (
        isinstance(timeout, bool) or not isinstance(timeout, (int, float))
    ):
        raise ValueError("timeout_seconds must be a number")
    return ProjectStatusIntegrationConfig(
        base_url=base_url,
        timeout_seconds=float(timeout) if timeout is not None else 5.0,
    )


class ProjectStatusTenantConnectionIntegrationFactory(
    EagerTenantConnectionIntegrationFactoryMixin,
):
    """Compose one Project Status integration from durable tenant connection config."""

    def __init__(
        self,
        *,
        http_client_factory: Callable[[ProjectStatusIntegrationConfig], Any] | None = None,
    ) -> None:
        self._http_client_factory = http_client_factory

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
    ) -> ProjectStatusIntegration:
        return self.create_integration_with_resolved_credential(
            tenant_id=tenant_id,
            connection_ref=connection_ref,
            provider_id=provider_id,
            integration_kind=integration_kind,
            credential_ref=credential_ref,
            resolved_credential=credential,
            secret_free_config=secret_free_config,
        )

    def create_integration_with_resolved_credential(
        self,
        *,
        tenant_id: str,
        connection_ref: str,
        provider_id: str,
        integration_kind: IntegrationCategory,
        credential_ref: str,
        resolved_credential: str,
        secret_free_config: Mapping[str, JsonValue],
    ) -> ProjectStatusIntegration:
        _require_nonblank(tenant_id, field_name="tenant_id")
        _require_nonblank(connection_ref, field_name="connection_ref")
        _require_nonblank(credential_ref, field_name="credential_ref")
        if provider_id != PROJECT_STATUS_PROVIDER_ID:
            raise ValueError("provider_id does not match project_status")
        if integration_kind is not IntegrationCategory.ISSUE_TRACKER:
            raise ValueError("integration_kind does not match issue_tracker")
        config = _parse_secret_free_config(secret_free_config)
        factory = self._http_client_factory or (
            lambda parsed: HttpxProjectStatusReadClient(config=parsed)
        )
        client = factory(config)
        return ProjectStatusIntegration.from_client(client, config=config)


__all__ = ["ProjectStatusTenantConnectionIntegrationFactory"]
