# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Hosted application lifecycle and shutdown port contracts."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from intergrax.hosting.contracts.public_data import validate_bounded_identifier


def _validate_timezone_aware_datetime(value: datetime, *, field_name: str) -> datetime:
    if value.tzinfo is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return value


class HostedApplicationLifecycleState(str, Enum):
    CREATED = "created"
    STARTING = "starting"
    READY = "ready"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"


class HostedApplicationLifecycleSnapshot(BaseModel):
    """Immutable lifecycle snapshot for hosting diagnostics and readiness."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    state: HostedApplicationLifecycleState
    accepting_new_work: bool
    shutdown_requested: bool
    last_transition_at: datetime
    reason_code: str = ""

    @field_validator("reason_code")
    @classmethod
    def _validate_reason_code(cls, value: str) -> str:
        if not value:
            return ""
        return validate_bounded_identifier(value, field_name="reason_code")

    @field_validator("last_transition_at")
    @classmethod
    def _validate_last_transition_at(cls, value: datetime) -> datetime:
        return _validate_timezone_aware_datetime(value, field_name="last_transition_at")


class HostedApplicationShutdownRequestSnapshot(BaseModel):
    """Immutable shutdown request snapshot exposed through context ports."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    reason_code: str
    requested_at: datetime
    deadline_at: datetime | None = None

    @field_validator("reason_code")
    @classmethod
    def _validate_reason_code(cls, value: str) -> str:
        return validate_bounded_identifier(value, field_name="reason_code")

    @field_validator("requested_at", "deadline_at")
    @classmethod
    def _validate_timestamps(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        return _validate_timezone_aware_datetime(value, field_name="timestamp")

    @model_validator(mode="after")
    def _validate_deadline_order(self) -> HostedApplicationShutdownRequestSnapshot:
        if self.deadline_at is not None and self.deadline_at < self.requested_at:
            raise ValueError("deadline_at must not be earlier than requested_at")
        return self


@runtime_checkable
class HostedApplicationLifecycleSnapshotProvider(Protocol):
    """Narrow lifecycle snapshot provider port for hosted application context."""

    def snapshot(self) -> HostedApplicationLifecycleSnapshot: ...


@runtime_checkable
class HostedApplicationShutdownCoordinator(Protocol):
    """Narrow shutdown request/token port for hosted application context."""

    def is_shutdown_requested(self) -> bool: ...

    def current_request(self) -> HostedApplicationShutdownRequestSnapshot | None: ...

    def request_shutdown(
        self,
        reason_code: str,
        *,
        deadline_at: datetime | None = None,
    ) -> HostedApplicationShutdownRequestSnapshot: ...

    async def wait_until_requested(self) -> HostedApplicationShutdownRequestSnapshot: ...
