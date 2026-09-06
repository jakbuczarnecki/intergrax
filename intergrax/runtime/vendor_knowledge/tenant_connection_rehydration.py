# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Restart rehydration of durable tenant connections into the runtime registry."""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.contracts.credential import supports_late_credential_resolution
from intergrax.integrations.contracts.secrets_store import SecretsStore
from intergrax.runtime.vendor_knowledge.connections import KnowledgeConnectionRegistry
from intergrax.runtime.vendor_knowledge.models import JsonValue
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.runtime.vendor_knowledge.errors import (
    VendorKnowledgeError,
    VendorKnowledgeErrorCode,
)
from intergrax.runtime.vendor_knowledge.tenant_connections import (
    SafeTenantConnectionV1,
    TenantConnection,
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
        credential_ref: str,
        credential: str,
        secret_free_config: Mapping[str, JsonValue],
    ) -> object:
        ...


class TenantConnectionRuntimeRegistryReconciler:
    """Ensure durable tenant connections are present in the instance-local registry."""

    def __init__(
        self,
        *,
        rehydrator: TenantConnectionRehydrator,
        connection_registry: KnowledgeConnectionRegistry,
    ) -> None:
        self._rehydrator = rehydrator
        self._connection_registry = connection_registry

    def ensure_registered(
        self,
        *,
        tenant_id: str,
        connection_ref: str,
        provider_id: str,
        integration_kind: IntegrationCategory,
    ) -> None:
        cleaned_tenant = tenant_id.strip()
        cleaned_ref = connection_ref.strip()
        cleaned_provider = provider_id.strip()
        if not cleaned_tenant:
            raise ValueError("tenant_id must be a non-empty string")
        if not cleaned_ref:
            raise ValueError("connection_ref must be a non-empty string")
        if not cleaned_provider:
            raise ValueError("provider_id must be a non-empty string")

        try:
            self._connection_registry.resolve(
                tenant_id=cleaned_tenant,
                connection_ref=cleaned_ref,
                provider_id=cleaned_provider,
                integration_kind=integration_kind,
            )
            return
        except VendorKnowledgeError as exc:
            if exc.code is not VendorKnowledgeErrorCode.INTEGRATION_NOT_FOUND:
                raise

        try:
            result = self._rehydrator.rehydrate_connection(
                tenant_id=cleaned_tenant,
                connection_ref=cleaned_ref,
            )
        except ValueError:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INTEGRATION_NOT_FOUND,
                safe_message="Requested connection is not registered for this tenant",
                provider_id=cleaned_provider,
                retryable=False,
            ) from None
        if result.status is not TenantConnectionRehydrationStatus.REGISTERED:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INTEGRATION_NOT_FOUND,
                safe_message="Requested connection is not registered for this tenant",
                provider_id=cleaned_provider,
                retryable=False,
            )

        self._connection_registry.resolve(
            tenant_id=cleaned_tenant,
            connection_ref=cleaned_ref,
            provider_id=cleaned_provider,
            integration_kind=integration_kind,
        )


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
            results.append(self._rehydrate_connection_record(connection))
        return tuple(results)

    def rehydrate_connection(
        self,
        *,
        tenant_id: str,
        connection_ref: str,
    ) -> TenantConnectionRehydrationResult:
        cleaned_tenant = tenant_id.strip()
        cleaned_ref = connection_ref.strip()
        if not cleaned_tenant:
            raise ValueError("tenant_id must be a non-empty string")
        if not cleaned_ref:
            raise ValueError("connection_ref must be a non-empty string")

        connection = self._repository.get(
            tenant_id=cleaned_tenant,
            connection_ref=cleaned_ref,
        )
        if connection is None:
            raise ValueError("connection_ref must reference an existing tenant connection")
        return self._rehydrate_connection_record(connection)

    def _rehydrate_connection_record(
        self,
        connection: TenantConnection,
    ) -> TenantConnectionRehydrationResult:
        safe = to_safe_tenant_connection(connection)
        status = connection.administrative_status

        if status is TenantConnectionAdministrativeStatus.DISABLED:
            return TenantConnectionRehydrationResult(
                connection=safe,
                status=TenantConnectionRehydrationStatus.SKIPPED_DISABLED,
            )

        if status is TenantConnectionAdministrativeStatus.REVOKED:
            return TenantConnectionRehydrationResult(
                connection=safe,
                status=TenantConnectionRehydrationStatus.SKIPPED_REVOKED,
            )

        credential: str | None = None
        if self._uses_late_credential_resolution(connection):
            credential = ""
        else:
            credential = self._resolve_secret(connection.credential_ref)
            if credential is None:
                return TenantConnectionRehydrationResult(
                    connection=safe,
                    status=TenantConnectionRehydrationStatus.UNAVAILABLE,
                    error_code="tenant_connection_secret_unavailable",
                )

        integration = self._construct_integration(connection, credential)
        if integration is None:
            return TenantConnectionRehydrationResult(
                connection=safe,
                status=TenantConnectionRehydrationStatus.UNAVAILABLE,
                error_code="tenant_connection_runtime_unavailable",
            )

        if not self._register_integration(connection, integration):
            return TenantConnectionRehydrationResult(
                connection=safe,
                status=TenantConnectionRehydrationStatus.UNAVAILABLE,
                error_code="tenant_connection_runtime_unavailable",
            )

        return TenantConnectionRehydrationResult(
            connection=safe,
            status=TenantConnectionRehydrationStatus.REGISTERED,
        )

    def _uses_late_credential_resolution(self, connection: TenantConnection) -> bool:
        factory = self._resolve_integration_factory(connection)
        if factory is None:
            return False
        return supports_late_credential_resolution(factory)

    def _resolve_integration_factory(self, connection: TenantConnection) -> object | None:
        factory = self._integration_factory
        if hasattr(factory, "factory_for"):
            resolved = factory.factory_for(
                provider_id=connection.provider_id,
                integration_kind=connection.integration_kind,
            )
            return resolved
        if supports_late_credential_resolution(factory):
            return factory
        return None

    def _resolve_secret(self, credential_ref: str) -> str | None:
        try:
            secret = self._secrets_store.get_secret(credential_ref)
        except Exception:
            return None
        if not isinstance(secret, str):
            return None
        if not secret.strip():
            return None
        return secret

    def _construct_integration(
        self,
        connection,
        credential: str | None,
    ) -> object | None:
        if credential is None:
            return None
        try:
            return self._integration_factory.create_integration(
                tenant_id=connection.tenant_id,
                connection_ref=connection.connection_ref,
                provider_id=connection.provider_id,
                integration_kind=connection.integration_kind,
                credential_ref=connection.credential_ref,
                credential=credential,
                secret_free_config=connection.validated_secret_free_config,
            )
        except Exception:
            return None

    def _register_integration(self, connection, integration: object) -> bool:
        try:
            self._connection_registry.refresh(
                tenant_id=connection.tenant_id,
                connection_ref=connection.connection_ref,
                provider_id=connection.provider_id,
                integration_kind=connection.integration_kind,
                integration=integration,
            )
        except Exception:
            return False
        return True
