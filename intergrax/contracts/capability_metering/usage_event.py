# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Source-qualified capability usage event contract (CAPABILITY-CATALOG-1 Stage 13)."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from intergrax.contracts.capability_catalog.identity import CapabilitySourceKind
from intergrax.contracts.capability_catalog.identity_key import CapabilityIdentityKey
from intergrax.contracts.capability_catalog.kind import CapabilityKind
from intergrax.contracts.capability_catalog.provenance import CapabilityProvenance
from intergrax.contracts.execution_identity import (
    AttemptId,
    EventId,
    RunId,
    TaskId,
    mint_event_id,
    validate_attempt_id,
    validate_event_id,
    validate_run_id,
    validate_task_id,
)

SCHEMA_CAPABILITY_USAGE_EVENT_V1: Final = "capability_usage_event.v1"


class CapabilityUsageKind(StrEnum):
    """Minimal execution semantics — not monetary units."""

    EXECUTION = "execution"
    DELEGATION = "delegation"


class CapabilityUsageOutcome(StrEnum):
    """Terminal invocation outcome — not price or billing state."""

    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class CapabilityUsageEvent(BaseModel):
    """Immutable usage evidence for downstream metering — not registry authority."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["capability_usage_event.v1"] = SCHEMA_CAPABILITY_USAGE_EVENT_V1
    event_id: EventId = Field(default_factory=mint_event_id)
    recorded_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    tenant_id: str
    identity: CapabilityIdentityKey
    provenance: CapabilityProvenance
    usage_kind: CapabilityUsageKind
    outcome: CapabilityUsageOutcome
    quantity: int = Field(default=1, ge=1)
    task_id: TaskId | None = None
    run_id: RunId | None = None
    attempt_id: AttemptId | None = None

    @field_validator("event_id", mode="before")
    @classmethod
    def _validate_event_id_field(cls, value: object) -> EventId:
        return validate_event_id(value)

    @field_validator("recorded_at")
    @classmethod
    def _validate_recorded_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("recorded_at must be timezone-aware")
        return value.astimezone(timezone.utc)

    @field_validator("tenant_id")
    @classmethod
    def _validate_tenant_id(cls, value: str) -> str:
        if type(value) is not str:
            raise TypeError("tenant_id must be str")
        if not value or not value.strip():
            raise ValueError("tenant_id must be non-empty and not whitespace-only")
        if value != value.strip():
            raise ValueError("tenant_id must not contain leading or trailing whitespace")
        return value

    @field_validator("task_id", mode="before")
    @classmethod
    def _validate_task_id_field(cls, value: object | None) -> TaskId | None:
        if value is None:
            return None
        return validate_task_id(value)

    @field_validator("run_id", mode="before")
    @classmethod
    def _validate_run_id_field(cls, value: object | None) -> RunId | None:
        if value is None:
            return None
        return validate_run_id(value)

    @field_validator("attempt_id", mode="before")
    @classmethod
    def _validate_attempt_id_field(cls, value: object | None) -> AttemptId | None:
        if value is None:
            return None
        return validate_attempt_id(value)

    @model_validator(mode="after")
    def _enforce_attribution_contract(self) -> CapabilityUsageEvent:
        if self.identity.kind is CapabilityKind.SKILL:
            raise ValueError(
                "CapabilityKind.SKILL cannot be a direct capability usage subject",
            )
        if self.identity.source_kind is CapabilitySourceKind.UNKNOWN:
            raise ValueError(
                "CapabilitySourceKind.UNKNOWN cannot satisfy source-qualified usage attribution",
            )
        if self.identity.source_id != self.provenance.source.source_id:
            raise ValueError(
                "identity.source_id must equal provenance.source.source_id",
            )
        if self.identity.source_kind != self.provenance.source.source_kind:
            raise ValueError(
                "identity.source_kind must equal provenance.source.source_kind",
            )
        return self


def build_capability_usage_event(
    *,
    tenant_id: str,
    identity: CapabilityIdentityKey,
    provenance: CapabilityProvenance,
    usage_kind: CapabilityUsageKind,
    outcome: CapabilityUsageOutcome,
    quantity: int = 1,
    task_id: TaskId | None = None,
    run_id: RunId | None = None,
    attempt_id: AttemptId | None = None,
    event_id: EventId | None = None,
    recorded_at: datetime | None = None,
) -> CapabilityUsageEvent:
    """Pure factory — all attribution inputs must be explicit; no catalog lookup."""
    payload: dict[str, object] = {
        "tenant_id": tenant_id,
        "identity": identity,
        "provenance": provenance,
        "usage_kind": usage_kind,
        "outcome": outcome,
        "quantity": quantity,
        "task_id": task_id,
        "run_id": run_id,
        "attempt_id": attempt_id,
    }
    if event_id is not None:
        payload["event_id"] = event_id
    if recorded_at is not None:
        payload["recorded_at"] = recorded_at
    return CapabilityUsageEvent.model_validate(payload)
