# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Hosted application context contracts and runtime ports (APP-HOST-1B)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, JsonValue, field_validator

from intergrax.hosting.contracts.events import HostedApplicationEvent
from intergrax.hosting.contracts.identity import normalize_application_id
from intergrax.hosting.contracts.lifecycle import (
    HostedApplicationLifecycleSnapshot,
    HostedApplicationLifecycleSnapshotProvider,
    HostedApplicationShutdownCoordinator,
)
from intergrax.hosting.contracts.profile import HostedApplicationProfilePublicView
from intergrax.hosting.contracts.public_data import (
    normalize_public_json_mapping,
    validate_instance_id,
)
from intergrax.hosting.services import HostedApplicationServiceRegistry

_PROFILE_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


def _validate_timezone_aware_datetime(value: datetime, *, field_name: str) -> datetime:
    if value.tzinfo is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return value


class HostedApplicationPaths(BaseModel):
    """Resolved hosted application filesystem paths."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    data_home: Path
    run_directory: Path


class HostedApplicationProcessIdentity(BaseModel):
    """Safe process identity exposed through hosted application context."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    process_id: int
    host_id: str | None = None
    user_scope_id: str | None = None
    started_at: datetime

    @field_validator("started_at")
    @classmethod
    def _validate_started_at(cls, value: datetime) -> datetime:
        return _validate_timezone_aware_datetime(value, field_name="started_at")


class HostedApplicationContextPublicView(BaseModel):
    """Safe public projection of a hosted application context."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    application_id: str
    instance_id: str
    profile_digest: str
    profile_spec_version: str
    process_identity: HostedApplicationProcessIdentity
    lifecycle: HostedApplicationLifecycleSnapshot
    closed: bool = False


@runtime_checkable
class HostedApplicationClock(Protocol):
    """Minimal clock port for hosted application extensions."""

    def now(self) -> datetime: ...


@runtime_checkable
class HostedApplicationLogger(Protocol):
    """Minimal structured logger port for hosted application extensions."""

    def debug(self, message: str, **fields: JsonValue) -> None: ...

    def info(self, message: str, **fields: JsonValue) -> None: ...

    def warning(self, message: str, **fields: JsonValue) -> None: ...

    def error(self, message: str, **fields: JsonValue) -> None: ...


@runtime_checkable
class HostedApplicationEventPublisher(Protocol):
    """Hosting event publication port."""

    async def publish(self, event: HostedApplicationEvent) -> None: ...


@dataclass(frozen=True, slots=True)
class HostedApplicationContext:
    """Instance-scoped hosted application runtime context contract."""

    application_id: str
    instance_id: str
    profile: HostedApplicationProfilePublicView
    profile_digest: str
    paths: HostedApplicationPaths
    process_identity: HostedApplicationProcessIdentity
    services: HostedApplicationServiceRegistry
    clock: HostedApplicationClock
    logger: HostedApplicationLogger
    event_publisher: HostedApplicationEventPublisher
    shutdown: HostedApplicationShutdownCoordinator
    lifecycle: HostedApplicationLifecycleSnapshotProvider
    _closed: bool = False

    def __post_init__(self) -> None:
        normalized_application_id = normalize_application_id(self.application_id)
        object.__setattr__(self, "application_id", normalized_application_id)
        object.__setattr__(self, "instance_id", validate_instance_id(self.instance_id))
        if not _PROFILE_DIGEST_RE.match(self.profile_digest):
            raise ValueError("profile_digest must match sha256:<64 lowercase hex>")
        if normalized_application_id != self.profile.identity.application_id:
            raise ValueError("application_id must match profile.identity.application_id")

    @property
    def closed(self) -> bool:
        return self._closed

    def public_view(self) -> HostedApplicationContextPublicView:
        return HostedApplicationContextPublicView(
            application_id=self.application_id,
            instance_id=self.instance_id,
            profile_digest=self.profile_digest,
            profile_spec_version=self.profile.spec_version,
            process_identity=self.process_identity,
            lifecycle=self.lifecycle.snapshot(),
            closed=self._closed,
        )

    def close(self) -> None:
        if self._closed:
            return
        object.__setattr__(self, "_closed", True)
        self.services.close()


def validate_logger_fields(fields: dict[str, JsonValue]) -> dict[str, JsonValue]:
    """Validate structured logger field payloads."""
    return normalize_public_json_mapping(fields)
