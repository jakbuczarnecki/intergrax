# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Hosted application hook contracts (APP-HOST-1C)."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from enum import Enum
from typing import TYPE_CHECKING, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic.json_schema import SkipJsonSchema

from intergrax.hosting.contracts.public_data import (
    derive_stable_callable_id,
    validate_bounded_identifier,
    validate_bounded_priority,
    validate_positive_bounded_seconds,
)
from intergrax.utils import attribute_access

if TYPE_CHECKING:
    from intergrax.hosting.contracts.context import HostedApplicationContext


class HostedApplicationHookCallback(Protocol):
    def __call__(
        self,
        context: HostedApplicationContext,
    ) -> None | Awaitable[None]: ...


HostedApplicationHookHandler = Callable[["HostedApplicationContext"], None | Awaitable[None]]


class HostedApplicationHookPoint(str, Enum):
    BEFORE_START = "before_start"
    BEFORE_READY = "before_ready"
    BEFORE_STOP = "before_stop"
    AFTER_START = "after_start"
    AFTER_READY = "after_ready"
    AFTER_STOP = "after_stop"
    ON_FAILURE = "on_failure"


class HostedApplicationHookMode(str, Enum):
    BLOCKING = "blocking"
    OBSERVER = "observer"


HOSTED_APPLICATION_HOOK_POINT_ORDER: tuple[HostedApplicationHookPoint, ...] = (
    HostedApplicationHookPoint.BEFORE_START,
    HostedApplicationHookPoint.BEFORE_READY,
    HostedApplicationHookPoint.BEFORE_STOP,
    HostedApplicationHookPoint.AFTER_START,
    HostedApplicationHookPoint.AFTER_READY,
    HostedApplicationHookPoint.AFTER_STOP,
    HostedApplicationHookPoint.ON_FAILURE,
)

_HOOK_POINT_ORDER = HOSTED_APPLICATION_HOOK_POINT_ORDER

_HOOK_POINT_MODES: dict[HostedApplicationHookPoint, HostedApplicationHookMode] = {
    HostedApplicationHookPoint.BEFORE_START: HostedApplicationHookMode.BLOCKING,
    HostedApplicationHookPoint.BEFORE_READY: HostedApplicationHookMode.BLOCKING,
    HostedApplicationHookPoint.BEFORE_STOP: HostedApplicationHookMode.BLOCKING,
    HostedApplicationHookPoint.AFTER_START: HostedApplicationHookMode.OBSERVER,
    HostedApplicationHookPoint.AFTER_READY: HostedApplicationHookMode.OBSERVER,
    HostedApplicationHookPoint.AFTER_STOP: HostedApplicationHookMode.OBSERVER,
    HostedApplicationHookPoint.ON_FAILURE: HostedApplicationHookMode.OBSERVER,
}


def hook_point_mode(point: HostedApplicationHookPoint) -> HostedApplicationHookMode:
    return _HOOK_POINT_MODES[point]


class HostedApplicationHookPublicDescriptor(BaseModel):
    """Public hook registration descriptor."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    hook_id: str
    handler_id: str
    hook_point: HostedApplicationHookPoint
    mode: HostedApplicationHookMode
    priority: int
    timeout_seconds: float | None
    source_id: str
    declaration_index: int


class HostedApplicationHook(BaseModel):
    """Runtime hook registration for a hosted application profile."""

    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    hook_id: str
    handler: SkipJsonSchema[HostedApplicationHookHandler] = Field(exclude=True, repr=False)
    handler_id: str | None = None
    priority: int = 0
    timeout_seconds: float | None = None
    source_id: str = "profile"

    @field_validator("hook_id")
    @classmethod
    def _validate_hook_id(cls, value: str) -> str:
        return validate_bounded_identifier(value, field_name="hook_id")

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

    @field_validator("timeout_seconds")
    @classmethod
    def _validate_timeout(cls, value: float | None) -> float | None:
        if value is None:
            return None
        return validate_positive_bounded_seconds(value, field_name="timeout_seconds")

    @model_validator(mode="after")
    def _resolve_handler_id(self) -> HostedApplicationHook:
        if self.handler_id is not None:
            return self
        derived_handler_id = derive_stable_callable_id(self.handler, field_name="handler_id")
        object.__setattr__(self, "handler_id", derived_handler_id)
        return self

    def public_descriptor(
        self,
        *,
        hook_point: HostedApplicationHookPoint,
        declaration_index: int,
    ) -> HostedApplicationHookPublicDescriptor:
        return HostedApplicationHookPublicDescriptor(
            hook_id=self.hook_id,
            handler_id=self.handler_id or "",
            hook_point=hook_point,
            mode=hook_point_mode(hook_point),
            priority=self.priority,
            timeout_seconds=self.timeout_seconds,
            source_id=self.source_id,
            declaration_index=declaration_index,
        )


class HostedApplicationHooks(BaseModel):
    """Author-facing hook registration collections."""

    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    before_start: tuple[HostedApplicationHook, ...] = ()
    before_ready: tuple[HostedApplicationHook, ...] = ()
    before_stop: tuple[HostedApplicationHook, ...] = ()
    after_start: tuple[HostedApplicationHook, ...] = ()
    after_ready: tuple[HostedApplicationHook, ...] = ()
    after_stop: tuple[HostedApplicationHook, ...] = ()
    on_failure: tuple[HostedApplicationHook, ...] = ()

    @model_validator(mode="after")
    def _reject_duplicate_hook_ids(self) -> HostedApplicationHooks:
        seen: set[str] = set()
        for point in _HOOK_POINT_ORDER:
            for hook in self.hooks_for_point(point):
                if hook.hook_id in seen:
                    raise ValueError(f"duplicate hook_id: {hook.hook_id}")
                seen.add(hook.hook_id)
        return self

    def hooks_for_point(self, point: HostedApplicationHookPoint) -> tuple[HostedApplicationHook, ...]:
        return attribute_access.optional(self, point.value)

    def flattened_public_descriptors(self) -> tuple[HostedApplicationHookPublicDescriptor, ...]:
        descriptors: list[HostedApplicationHookPublicDescriptor] = []
        for point in _HOOK_POINT_ORDER:
            for index, hook in enumerate(self.hooks_for_point(point)):
                descriptors.append(
                    hook.public_descriptor(hook_point=point, declaration_index=index),
                )
        descriptors.sort(
            key=lambda item: (
                _HOOK_POINT_ORDER.index(item.hook_point),
                item.priority,
                item.source_id,
                item.declaration_index,
            )
        )
        return tuple(descriptors)
