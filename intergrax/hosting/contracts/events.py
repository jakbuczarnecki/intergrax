# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Typed hosting event contracts (APP-HOST-3A)."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from enum import Enum
from typing import TYPE_CHECKING
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic import JsonValue
from pydantic.json_schema import SkipJsonSchema

from intergrax.contracts.event_severity import EventSeverity
from intergrax.hosting.contracts.lifecycle import HostedApplicationLifecycleState
from intergrax.hosting.contracts.public_data import (
    derive_stable_callable_id,
    normalize_public_json_mapping,
    validate_bounded_identifier,
    validate_bounded_priority,
)

if TYPE_CHECKING:
    pass

HOSTED_APPLICATION_EVENT_SCHEMA_ID = "intergrax.hosting.event"
HOSTED_APPLICATION_EVENT_SCHEMA_VERSION = "1.0"

HostedApplicationEventHandler = Callable[..., None | Awaitable[None]]


class HostedApplicationEventType(str, Enum):
    APPLICATION_STARTING = "hosting.application.starting"
    APPLICATION_STARTED = "hosting.application.started"
    APPLICATION_READY = "hosting.application.ready"
    APPLICATION_STOPPING = "hosting.application.stopping"
    APPLICATION_STOPPED = "hosting.application.stopped"
    APPLICATION_FAILED = "hosting.application.failed"

    COMPONENT_STARTING = "hosting.component.starting"
    COMPONENT_STARTED = "hosting.component.started"
    COMPONENT_HEALTH_CHANGED = "hosting.component.health_changed"
    COMPONENT_STOPPING = "hosting.component.stopping"
    COMPONENT_STOPPED = "hosting.component.stopped"
    COMPONENT_FAILED = "hosting.component.failed"

    INSTANCE_ACQUIRED = "hosting.instance.acquired"
    INSTANCE_REJECTED = "hosting.instance.rejected"
    INSTANCE_STALE_RECOVERED = "hosting.instance.stale_recovered"
    INSTANCE_RELEASED = "hosting.instance.released"

    RESTART_REQUESTED = "hosting.restart.requested"
    RESTART_SCHEDULED = "hosting.restart.scheduled"
    RESTART_STARTED = "hosting.restart.started"
    RESTART_EXHAUSTED = "hosting.restart.exhausted"

    HOOK_STARTED = "hosting.hook.started"
    HOOK_COMPLETED = "hosting.hook.completed"
    HOOK_FAILED = "hosting.hook.failed"

    PLUGIN_LOADED = "hosting.plugin.loaded"
    PLUGIN_FAILED = "hosting.plugin.failed"


class HostedApplicationEvent(BaseModel):
    """Versioned hosting event envelope."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_id: str = HOSTED_APPLICATION_EVENT_SCHEMA_ID
    schema_version: str = HOSTED_APPLICATION_EVENT_SCHEMA_VERSION
    event_id: str = Field(default_factory=lambda: str(uuid4()))
    event_type: HostedApplicationEventType
    occurred_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    application_id: str
    instance_id: str
    lifecycle_state: HostedApplicationLifecycleState
    severity: EventSeverity = EventSeverity.INFO
    correlation_id: str = ""
    causation_id: str = ""
    payload: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("schema_id")
    @classmethod
    def _validate_schema_id(cls, value: str) -> str:
        if value != HOSTED_APPLICATION_EVENT_SCHEMA_ID:
            raise ValueError("invalid hosting event schema_id")
        return value

    @field_validator("schema_version")
    @classmethod
    def _validate_schema_version(cls, value: str) -> str:
        if value != HOSTED_APPLICATION_EVENT_SCHEMA_VERSION:
            raise ValueError("invalid hosting event schema_version")
        return value

    @field_validator("event_id")
    @classmethod
    def _validate_event_id(cls, value: str) -> str:
        if not value:
            return ""
        return validate_bounded_identifier(value, field_name="event_id")

    @field_validator("correlation_id", "causation_id")
    @classmethod
    def _validate_correlation_identifiers(cls, value: str) -> str:
        if not value:
            return ""
        return validate_bounded_identifier(value, field_name="correlation_id")

    @field_validator("occurred_at")
    @classmethod
    def _validate_occurred_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("occurred_at must be timezone-aware")
        return value

    @field_validator("payload")
    @classmethod
    def _validate_payload(cls, value: dict[str, JsonValue]) -> dict[str, JsonValue]:
        return normalize_public_json_mapping(value)


class HostedApplicationEventSubscriptionPublicDescriptor(BaseModel):
    """Public event subscription descriptor."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    subscription_id: str
    event_types: tuple[HostedApplicationEventType, ...]
    handler_id: str
    source_id: str
    priority: int
    declaration_index: int


class HostedApplicationEventSubscription(BaseModel):
    """Runtime event subscription registration for a hosted application profile."""

    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    subscription_id: str
    event_types: tuple[HostedApplicationEventType, ...]
    handler: SkipJsonSchema[HostedApplicationEventHandler] = Field(exclude=True, repr=False)
    handler_id: str | None = None
    source_id: str = "profile"
    priority: int = 0

    @field_validator("subscription_id")
    @classmethod
    def _validate_subscription_id(cls, value: str) -> str:
        return validate_bounded_identifier(value, field_name="subscription_id")

    @field_validator("handler_id")
    @classmethod
    def _validate_handler_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return validate_bounded_identifier(value, field_name="handler_id")

    @field_validator("source_id")
    @classmethod
    def _validate_source_id(cls, value: str) -> str:
        return validate_bounded_identifier(value, field_name="source_id")

    @field_validator("priority")
    @classmethod
    def _validate_priority(cls, value: int) -> int:
        return validate_bounded_priority(value)

    @field_validator("event_types")
    @classmethod
    def _validate_event_types(
        cls,
        value: tuple[HostedApplicationEventType, ...],
    ) -> tuple[HostedApplicationEventType, ...]:
        if not value:
            raise ValueError("event_types must not be empty")
        return value

    @model_validator(mode="after")
    def _resolve_handler_id(self) -> HostedApplicationEventSubscription:
        if self.handler_id is not None:
            return self
        derived_handler_id = derive_stable_callable_id(self.handler, field_name="handler_id")
        object.__setattr__(self, "handler_id", derived_handler_id)
        return self

    def public_descriptor(self, *, declaration_index: int) -> HostedApplicationEventSubscriptionPublicDescriptor:
        return HostedApplicationEventSubscriptionPublicDescriptor(
            subscription_id=self.subscription_id,
            event_types=self.event_types,
            handler_id=self.handler_id or "",
            source_id=self.source_id,
            priority=self.priority,
            declaration_index=declaration_index,
        )
