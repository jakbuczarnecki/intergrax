# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Restart rehydration of durable tenant connections into the runtime registry."""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.contracts.secrets_store import SecretsStore
from intergrax.runtime.vendor_knowledge.connections import KnowledgeConnectionRegistry
from intergrax.runtime.vendor_knowledge.models import JsonValue
from intergrax.runtime.vendor_knowledge.tenant_connections import (
    SafeTenantConnectionV1,
    TenantConnectionAdministrativeStatus,
    TenantConnectionRepository,
    to_safe_tenant_connection,
)


class TenantConnectionRehydrationStatus(StrEnum):
    REGISTERED = "registered"
    SKIPPED_DISABLED = "skipped_disabled"
    SKIPPED_REVOKED = "skipped_revoked"
    UNAVAILABLE = "unavailable"


class TenantConnectionRehydrationResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    connection: SafeTenantConnectionV1
    status: TenantConnectionRehydrationStatus
    error_code: str | None = None


@runtime_checkable
class TenantConnectionIntegrationFactory(Protocol):
    def create_integration(
        self,
        *,
        tenant_id: str,
        connection_ref: str,
        provider_id: str,
        integration_kind: IntegrationCategory,
        credential: str,
        secret_free_config: Mapping[str, JsonValue],
    ) -> object:
        ...


class TenantConnectionRehydrator:
    """Reconstruct runtime connection registrations from durable tenant catalog."""

    def __init__(
        self,
        *,
        repository: TenantConnectionRepository,
        secrets_store: SecretsStore,
        integration_factory: TenantConnectionIntegrationFactory,
        connection_registry: KnowledgeConnectionRegistry,
    ) -> None:
        self._repository = repository
        self._secrets_store = secrets_store
        self._integration_factory = integration_factory
        self._connection_registry = connection_registry

    def rehydrate_tenant(
        self,
        *,
        tenant_id: str,
        limit: int = 1000,
    ) -> tuple[TenantConnectionRehydrationResult, ...]:
        cleaned_tenant = tenant_id.strip()
        if not cleaned_tenant:
            raise ValueError("tenant_id must be a non-empty string")

        connections = self._repository.list(tenant_id=cleaned_tenant, limit=limit)
        results: list[TenantConnectionRehydrationResult] = []
        for connection in connections:
            safe = to_safe_tenant_connection(connection)
            status = connection.administrative_status

            if status is TenantConnectionAdministrativeStatus.DISABLED:
                results.append(
                    TenantConnectionRehydrationResult(
                        connection=safe,
                        status=TenantConnectionRehydrationStatus.SKIPPED_DISABLED,
                    )
                )
                continue

            if status is TenantConnectionAdministrativeStatus.REVOKED:
                results.append(
                    TenantConnectionRehydrationResult(
                        connection=safe,
                        status=TenantConnectionRehydrationStatus.SKIPPED_REVOKED,
                    )
                )
                continue

            credential = self._resolve_secret(connection.credential_ref)
            if credential is None:
                results.append(
                    TenantConnectionRehydrationResult(
                        connection=safe,
                        status=TenantConnectionRehydrationStatus.UNAVAILABLE,
                        error_code="tenant_connection_secret_unavailable",
                    )
                )
                continue

            integration = self._construct_integration(connection, credential)
            if integration is None:
                results.append(
                    TenantConnectionRehydrationResult(
                        connection=safe,
                        status=TenantConnectionRehydrationStatus.UNAVAILABLE,
                        error_code="tenant_connection_runtime_unavailable",
                    )
                )
                continue

            if not self._register_integration(connection, integration):
                results.append(
                    TenantConnectionRehydrationResult(
                        connection=safe,
                        status=TenantConnectionRehydrationStatus.UNAVAILABLE,
                        error_code="tenant_connection_runtime_unavailable",
                    )
                )
                continue

            results.append(
                TenantConnectionRehydrationResult(
                    connection=safe,
                    status=TenantConnectionRehydrationStatus.REGISTERED,
                )
            )
        return tuple(results)

    def _resolve_secret(self, credential_ref: str) -> str | None:
        try:
            secret = self._secrets_store.get_secret(credential_ref)
        except Exception:
            return None
        cleaned = secret.strip()
        if not cleaned:
            return None
        return cleaned

    def _construct_integration(
        self,
        connection,
        credential: str,
    ) -> object | None:
        try:
            return self._integration_factory.create_integration(
                tenant_id=connection.tenant_id,
                connection_ref=connection.connection_ref,
                provider_id=connection.provider_id,
                integration_kind=connection.integration_kind,
                credential=credential,
                secret_free_config=connection.validated_secret_free_config,
            )
        except Exception:
            return None

    def _register_integration(self, connection, integration: object) -> bool:
        try:
            self._connection_registry.register(
                tenant_id=connection.tenant_id,
                connection_ref=connection.connection_ref,
                provider_id=connection.provider_id,
                integration_kind=connection.integration_kind,
                integration=integration,
            )
        except Exception:
            return False
        return True
