# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Durable tenant-scoped connection catalog for Vendor Knowledge."""

from __future__ import annotations

import re
from collections.abc import Mapping
from datetime import datetime
from enum import StrEnum
from typing import Protocol, runtime_checkable
from urllib.parse import urlparse

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator,
    model_validator,
)

from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.runtime.vendor_knowledge.models import JsonValue

_CONNECTION_REF_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")

_FORBIDDEN_EXACT_KEYS: frozenset[str] = frozenset(
    {
        "password",
        "passphrase",
        "secret",
        "client_secret",
        "access_token",
        "refresh_token",
        "id_token",
        "api_key",
        "authorization",
        "authorization_header",
        "private_key",
    }
)
_FORBIDDEN_SUFFIXES: tuple[str, ...] = (
    "_password",
    "_passphrase",
    "_client_secret",
    "_access_token",
    "_refresh_token",
    "_api_key",
    "_private_key",
)


class TenantConnectionAdministrativeStatus(StrEnum):
    ACTIVE = "active"
    DISABLED = "disabled"
    REVOKED = "revoked"


class TenantConnectionAlreadyExists(Exception):
    """Connection already exists for the requested tenant-scoped identity."""


class TenantConnectionNotFound(Exception):
    """Connection was not found for the requested tenant-scoped identity."""


class TenantConnectionVersionConflict(Exception):
    """Optimistic configuration version conflict."""


class TenantConnectionInvalidState(Exception):
    """Connection state or lifecycle contract violation."""


class TenantConnectionCorruptRecord(Exception):
    """Durable connection record is corrupt or inconsistent."""


def _require_non_empty(value: str, *, field_name: str, max_length: int | None = None) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field_name} must be a non-empty string")
    if max_length is not None and len(cleaned) > max_length:
        raise ValueError(f"{field_name} exceeds maximum length {max_length}")
    return cleaned


def _is_forbidden_secret_key(key: str) -> bool:
    normalized = key.strip().lower()
    if normalized in _FORBIDDEN_EXACT_KEYS:
        return True
    return any(normalized.endswith(suffix) for suffix in _FORBIDDEN_SUFFIXES)


def _url_has_embedded_credentials(value: str) -> bool:
    parsed = urlparse(value.strip())
    if not parsed.scheme or not parsed.hostname:
        return False
    return bool(parsed.username or parsed.password)


def _assert_secret_free_config(
    value: Mapping[str, JsonValue],
    *,
    field_name: str,
) -> dict[str, JsonValue]:
    def _walk(node: JsonValue, path: str) -> None:
        if isinstance(node, dict):
            for key, child in node.items():
                child_path = f"{path}.{key}" if path else key
                if _is_forbidden_secret_key(key):
                    raise ValueError(
                        f"{field_name} contains forbidden secret-bearing key at {child_path}"
                    )
                _walk(child, child_path)
        elif isinstance(node, list):
            for index, child in enumerate(node):
                _walk(child, f"{path}[{index}]")
        elif isinstance(node, str) and _url_has_embedded_credentials(node):
            raise ValueError(
                f"{field_name} contains credential-bearing URL at {path}"
            )

    result = dict(value)
    _walk(result, "")
    return result


def _assert_utc_aware(value: datetime, *, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be a timezone-aware UTC datetime")
    return value


class TenantConnection(BaseModel):
    """Durable tenant-scoped integration connection (secret-free configuration)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    connection_ref: str
    tenant_id: str
    provider_id: str
    integration_kind: IntegrationCategory
    safe_display_name: str
    administrative_status: TenantConnectionAdministrativeStatus
    credential_ref: str
    validated_secret_free_config: Mapping[str, JsonValue]
    configuration_version: int = Field(ge=1)
    created_at: datetime
    updated_at: datetime
    connected_principal_ref: str | None = None

    @field_validator("connection_ref")
    @classmethod
    def _valid_connection_ref(cls, value: str) -> str:
        cleaned = value.strip()
        if not _CONNECTION_REF_RE.fullmatch(cleaned):
            raise ValueError("connection_ref format is invalid")
        return cleaned

    @field_validator("tenant_id")
    @classmethod
    def _valid_tenant_id(cls, value: str) -> str:
        return _require_non_empty(value, field_name="tenant_id", max_length=128)

    @field_validator("provider_id")
    @classmethod
    def _valid_provider_id(cls, value: str) -> str:
        return _require_non_empty(value, field_name="provider_id", max_length=64)

    @field_validator("safe_display_name")
    @classmethod
    def _valid_display_name(cls, value: str) -> str:
        return _require_non_empty(value, field_name="safe_display_name", max_length=256)

    @field_validator("credential_ref")
    @classmethod
    def _valid_credential_ref(cls, value: str) -> str:
        return _require_non_empty(value, field_name="credential_ref", max_length=512)

    @field_validator("connected_principal_ref")
    @classmethod
    def _optional_principal_ref(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _require_non_empty(
            value,
            field_name="connected_principal_ref",
            max_length=256,
        )

    @field_validator("created_at", "updated_at")
    @classmethod
    def _utc_timestamps(cls, value: datetime, info: ValidationInfo) -> datetime:
        field_name = info.field_name or "timestamp"
        return _assert_utc_aware(value, field_name=field_name)

    @field_validator("validated_secret_free_config")
    @classmethod
    def _secret_free_config(
        cls,
        value: Mapping[str, JsonValue],
    ) -> dict[str, JsonValue]:
        return _assert_secret_free_config(value, field_name="validated_secret_free_config")

    @model_validator(mode="after")
    def _updated_at_not_before_created(self) -> TenantConnection:
        if self.updated_at < self.created_at:
            raise ValueError("updated_at must be greater than or equal to created_at")
        return self


class SafeTenantConnectionV1(BaseModel):
    """Safe projection of a durable tenant connection (no secrets or credential refs)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    connection_ref: str
    tenant_id: str
    provider_id: str
    integration_kind: IntegrationCategory
    safe_display_name: str
    administrative_status: TenantConnectionAdministrativeStatus
    configuration_version: int
    connected_principal_ref: str | None
    created_at: datetime
    updated_at: datetime


def to_safe_tenant_connection(connection: TenantConnection) -> SafeTenantConnectionV1:
    return SafeTenantConnectionV1(
        connection_ref=connection.connection_ref,
        tenant_id=connection.tenant_id,
        provider_id=connection.provider_id,
        integration_kind=connection.integration_kind,
        safe_display_name=connection.safe_display_name,
        administrative_status=connection.administrative_status,
        configuration_version=connection.configuration_version,
        connected_principal_ref=connection.connected_principal_ref,
        created_at=connection.created_at,
        updated_at=connection.updated_at,
    )


_ALLOWED_STATUS_TRANSITIONS: dict[
    TenantConnectionAdministrativeStatus,
    frozenset[TenantConnectionAdministrativeStatus],
] = {
    TenantConnectionAdministrativeStatus.ACTIVE: frozenset(
        {
            TenantConnectionAdministrativeStatus.ACTIVE,
            TenantConnectionAdministrativeStatus.DISABLED,
            TenantConnectionAdministrativeStatus.REVOKED,
        }
    ),
    TenantConnectionAdministrativeStatus.DISABLED: frozenset(
        {
            TenantConnectionAdministrativeStatus.DISABLED,
            TenantConnectionAdministrativeStatus.ACTIVE,
            TenantConnectionAdministrativeStatus.REVOKED,
        }
    ),
    TenantConnectionAdministrativeStatus.REVOKED: frozenset(),
}


@runtime_checkable
class TenantConnectionRepository(Protocol):
    def create(self, connection: TenantConnection) -> None:
        ...

    def get(
        self,
        *,
        tenant_id: str,
        connection_ref: str,
    ) -> TenantConnection | None:
        ...

    def update(
        self,
        connection: TenantConnection,
        *,
        expected_configuration_version: int,
    ) -> None:
        ...

    def list(
        self,
        *,
        tenant_id: str,
        limit: int = 100,
        administrative_status: TenantConnectionAdministrativeStatus | None = None,
    ) -> tuple[TenantConnection, ...]:
        ...


class TenantConnectionService:
    """Tenant-scoped administrative lifecycle for durable connections."""

    def __init__(
        self,
        *,
        tenant_id: str,
        repository: TenantConnectionRepository,
    ) -> None:
        cleaned_tenant = str(tenant_id).strip()
        if not cleaned_tenant:
            raise ValueError("tenant_id must be a non-empty string")
        self._tenant_id = cleaned_tenant
        self._repository = repository

    def create(self, connection: TenantConnection) -> TenantConnection:
        self._assert_service_tenant(connection.tenant_id)
        if connection.configuration_version != 1:
            raise TenantConnectionInvalidState(
                "new connection configuration_version must be 1"
            )
        if connection.administrative_status is not TenantConnectionAdministrativeStatus.ACTIVE:
            raise TenantConnectionInvalidState(
                "new connection administrative_status must be ACTIVE"
            )
        if connection.created_at != connection.updated_at:
            raise TenantConnectionInvalidState(
                "new connection created_at must equal updated_at"
            )
        self._repository.create(connection)
        return connection

    def get(self, connection_ref: str) -> TenantConnection:
        cleaned_ref = _require_non_empty(connection_ref, field_name="connection_ref")
        connection = self._repository.get(
            tenant_id=self._tenant_id,
            connection_ref=cleaned_ref,
        )
        if connection is None:
            raise TenantConnectionNotFound("tenant connection was not found")
        self._assert_service_tenant(connection.tenant_id)
        return connection

    def list(
        self,
        *,
        limit: int = 100,
        administrative_status: TenantConnectionAdministrativeStatus | None = None,
    ) -> tuple[TenantConnection, ...]:
        connections = self._repository.list(
            tenant_id=self._tenant_id,
            limit=limit,
            administrative_status=administrative_status,
        )
        for connection in connections:
            self._assert_service_tenant(connection.tenant_id)
        return connections

    def update(
        self,
        connection: TenantConnection,
        *,
        expected_configuration_version: int,
    ) -> TenantConnection:
        self._assert_service_tenant(connection.tenant_id)
        current = self.get(connection.connection_ref)
        if current.administrative_status is TenantConnectionAdministrativeStatus.REVOKED:
            raise TenantConnectionInvalidState("revoked connection cannot be updated")
        if current.configuration_version != expected_configuration_version:
            raise TenantConnectionVersionConflict(
                "tenant connection configuration version conflict"
            )
        if connection.configuration_version != current.configuration_version + 1:
            raise TenantConnectionVersionConflict(
                "tenant connection configuration version conflict"
            )
        if connection.updated_at <= current.updated_at:
            raise TenantConnectionInvalidState(
                "updated_at must be greater than the current record updated_at"
            )
        self._assert_immutable_fields(current, connection)
        allowed = _ALLOWED_STATUS_TRANSITIONS[current.administrative_status]
        if connection.administrative_status not in allowed:
            raise TenantConnectionInvalidState(
                "tenant connection administrative status transition is not allowed"
            )
        self._repository.update(
            connection,
            expected_configuration_version=expected_configuration_version,
        )
        return connection

    def get_safe(self, connection_ref: str) -> SafeTenantConnectionV1:
        return to_safe_tenant_connection(self.get(connection_ref))

    def list_safe(
        self,
        *,
        limit: int = 100,
        administrative_status: TenantConnectionAdministrativeStatus | None = None,
    ) -> tuple[SafeTenantConnectionV1, ...]:
        return tuple(
            to_safe_tenant_connection(connection)
            for connection in self.list(
                limit=limit,
                administrative_status=administrative_status,
            )
        )

    def _assert_service_tenant(self, tenant_id: str) -> None:
        if tenant_id != self._tenant_id:
            raise TenantConnectionInvalidState(
                "tenant connection tenant does not match the service tenant"
            )

    def _assert_immutable_fields(
        self,
        current: TenantConnection,
        replacement: TenantConnection,
    ) -> None:
        if (
            current.connection_ref != replacement.connection_ref
            or current.tenant_id != replacement.tenant_id
            or current.provider_id != replacement.provider_id
            or current.integration_kind != replacement.integration_kind
            or current.created_at != replacement.created_at
        ):
            raise TenantConnectionInvalidState(
                "tenant connection identity fields are immutable"
            )

