# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Hosted application component contracts (APP-HOST-1D)."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic.json_schema import SkipJsonSchema

from intergrax.hosting.contracts.policies import ComponentFailureAction
from intergrax.hosting.contracts.public_data import (
    derive_stable_type_id,
    validate_bounded_identifier,
    validate_positive_bounded_seconds,
)


def _validate_timezone_aware_datetime(value: datetime, *, field_name: str) -> datetime:
    if value.tzinfo is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return value


def _validate_bounded_safe_message(value: str, *, field_name: str, max_length: int = 512) -> str:
    if not value:
        return ""
    if len(value) > max_length:
        raise ValueError(f"{field_name} must be at most {max_length} characters")
    if any(ord(character) < 32 for character in value):
        raise ValueError(f"{field_name} must not contain control characters")
    return value

if TYPE_CHECKING:
    from intergrax.hosting.contracts.context import HostedApplicationContext


class HostedApplicationComponentState(str, Enum):
    CREATED = "created"
    STARTING = "starting"
    READY = "ready"
    DEGRADED = "degraded"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"


class HostedApplicationComponentHealth(BaseModel):
    """Safe component health snapshot."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    component_id: str
    enabled: bool
    required: bool
    state: HostedApplicationComponentState
    healthy: bool
    ready: bool
    detail_code: str = ""
    safe_message: str = ""
    last_transition_at: datetime | None = None
    last_check_at: datetime | None = None

    @field_validator("component_id")
    @classmethod
    def _validate_component_id_field(cls, value: str) -> str:
        return validate_bounded_identifier(value, field_name="component_id")

    @field_validator("detail_code")
    @classmethod
    def _validate_detail_code(cls, value: str) -> str:
        if not value:
            return ""
        return validate_bounded_identifier(value, field_name="detail_code")

    @field_validator("safe_message")
    @classmethod
    def _validate_safe_message(cls, value: str) -> str:
        return _validate_bounded_safe_message(value, field_name="safe_message")

    @field_validator("last_transition_at", "last_check_at")
    @classmethod
    def _validate_timestamps(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        return _validate_timezone_aware_datetime(value, field_name="timestamp")


@runtime_checkable
class HostedApplicationComponent(Protocol):
    """Runtime component lifecycle protocol for hosted applications."""

    @property
    def component_id(self) -> str: ...

    async def start(self, context: HostedApplicationContext) -> None: ...

    async def stop(self, context: HostedApplicationContext) -> None: ...

    async def health(self, context: HostedApplicationContext) -> HostedApplicationComponentHealth: ...


class HostedApplicationComponentPublicDescriptor(BaseModel):
    """Public component registration descriptor."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    component_id: str
    component_type_id: str
    enabled: bool
    required: bool
    dependencies: tuple[str, ...]
    start_timeout_seconds: float
    stop_timeout_seconds: float
    health_timeout_seconds: float
    failure_action: ComponentFailureAction
    declaration_index: int


class HostedApplicationComponentRegistration(BaseModel):
    """Component registration descriptor for a hosted application profile."""

    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    component: SkipJsonSchema[object] = Field(exclude=True, repr=False)
    component_id: str | None = None
    component_type_id: str | None = None
    enabled: bool = True
    required: bool = False
    dependencies: tuple[str, ...] = ()
    start_timeout_seconds: float = 30.0
    stop_timeout_seconds: float = 30.0
    health_timeout_seconds: float = 5.0
    failure_action: ComponentFailureAction | None = None

    @field_validator("component_id")
    @classmethod
    def _validate_component_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return validate_bounded_identifier(value, field_name="component_id")

    @field_validator("component_type_id")
    @classmethod
    def _validate_component_type_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return validate_bounded_identifier(value, field_name="component_type_id")

    @field_validator("dependencies")
    @classmethod
    def _validate_dependencies(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        seen: set[str] = set()
        for dependency in value:
            validated = validate_bounded_identifier(dependency, field_name="dependency")
            if validated in seen:
                raise ValueError(f"duplicate dependency: {validated}")
            seen.add(validated)
        return value

    @field_validator(
        "start_timeout_seconds",
        "stop_timeout_seconds",
        "health_timeout_seconds",
    )
    @classmethod
    def _validate_timeouts(cls, value: float) -> float:
        return validate_positive_bounded_seconds(value, field_name="component_timeout_seconds")

    @model_validator(mode="after")
    def _resolve_ids_and_validate(self) -> HostedApplicationComponentRegistration:
        if not isinstance(self.component, HostedApplicationComponent):
            raise ValueError("component must implement HostedApplicationComponent")
        resolved_component_id = self.component_id or self.component.component_id
        if not resolved_component_id:
            raise ValueError("component_id must not be empty")
        resolved_component_id = validate_bounded_identifier(
            resolved_component_id,
            field_name="component_id",
        )
        object.__setattr__(self, "component_id", resolved_component_id)

        if self.component_type_id is None:
            derived_type_id = derive_stable_type_id(
                self.component,
                field_name="component_type_id",
            )
            object.__setattr__(self, "component_type_id", derived_type_id)

        if self.required and not self.enabled:
            raise ValueError("required component cannot be disabled")

        if resolved_component_id in self.dependencies:
            raise ValueError("component cannot depend on itself")

        if self.failure_action is None:
            object.__setattr__(
                self,
                "failure_action",
                ComponentFailureAction.FAIL_HOST
                if self.required
                else ComponentFailureAction.MARK_DEGRADED,
            )
        return self

    def public_descriptor(self, *, declaration_index: int) -> HostedApplicationComponentPublicDescriptor:
        return HostedApplicationComponentPublicDescriptor(
            component_id=self.component_id or "",
            component_type_id=self.component_type_id or "",
            enabled=self.enabled,
            required=self.required,
            dependencies=self.dependencies,
            start_timeout_seconds=self.start_timeout_seconds,
            stop_timeout_seconds=self.stop_timeout_seconds,
            health_timeout_seconds=self.health_timeout_seconds,
            failure_action=self.failure_action or ComponentFailureAction.MARK_DEGRADED,
            declaration_index=declaration_index,
        )
