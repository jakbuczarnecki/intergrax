# © Artur Czarnecki. All rights reserved.

"""Planning and resolution models for conversational tenant connection journeys."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class TenantConnectionPlanningProviderV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    provider_id: str = Field(min_length=1, max_length=64)
    safe_display_name: str = Field(min_length=1, max_length=256)
    auth_mode: str = Field(min_length=1, max_length=64)
    qualification: str = Field(min_length=1, max_length=64)


class TenantConnectionPlanningConnectionV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    connection_ref: str = Field(min_length=1, max_length=128)
    provider_id: str = Field(min_length=1, max_length=64)
    safe_display_name: str = Field(min_length=1, max_length=256)
    administrative_status: str = Field(min_length=1, max_length=32)
    connected_principal_ref: str | None = Field(default=None, max_length=256)
    configuration_version: int = Field(ge=0)


class TenantConnectionPendingManualAuthorizationV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    authorization_transaction_ref: str = Field(min_length=1, max_length=128)
    provider_id: str = Field(min_length=1, max_length=64)


class TenantConnectionPlanningSnapshotV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    providers: tuple[TenantConnectionPlanningProviderV1, ...] = ()
    connections: tuple[TenantConnectionPlanningConnectionV1, ...] = ()
    pending_manual_authorization: TenantConnectionPendingManualAuthorizationV1 | None = None


THREAD_MEMORY_CREDENTIAL_REDACTION = "[credential submission redacted]"


__all__ = [
    "TenantConnectionPendingManualAuthorizationV1",
    "TenantConnectionPlanningConnectionV1",
    "TenantConnectionPlanningProviderV1",
    "TenantConnectionPlanningSnapshotV1",
    "THREAD_MEMORY_CREDENTIAL_REDACTION",
]
