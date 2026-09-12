# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Namespaced preventive action type registry (PREVENTIVE R7)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PreventiveActionTypeDescriptor:
    """Versioned, namespaced action type — plugin catalog entry."""

    namespace: str
    name: str
    version: str
    description: str = ""

    @property
    def qualified_id(self) -> str:
        return f"{self.namespace}.{self.name}@{self.version}"

    def __post_init__(self) -> None:
        for field_name, value in (
            ("namespace", self.namespace),
            ("name", self.name),
            ("version", self.version),
        ):
            if not value.strip():
                raise ValueError(f"{field_name} must be non-empty")


class PreventiveActionType:
    """Built-in catalog — extend via registry without hardcoded tenant logic."""

    RESOURCE_THROTTLE = PreventiveActionTypeDescriptor(
        namespace="resource",
        name="throttle",
        version="1",
        description="Reduce throughput on a scoped resource",
    )
    INTEGRATION_PAUSE = PreventiveActionTypeDescriptor(
        namespace="integration",
        name="pause",
        version="1",
        description="Pause an integration connector",
    )
    CONFIGURATION_UPDATE = PreventiveActionTypeDescriptor(
        namespace="configuration",
        name="update",
        version="1",
        description="Apply a bounded configuration change",
    )
    HUMAN_REVIEW = PreventiveActionTypeDescriptor(
        namespace="human",
        name="review",
        version="1",
        description="Escalate to human review without mutation",
    )
    CAPACITY_SCALE = PreventiveActionTypeDescriptor(
        namespace="capacity",
        name="scale",
        version="1",
        description="Scale capacity for a scoped workload",
    )


_BUILTIN_TYPES: tuple[PreventiveActionTypeDescriptor, ...] = (
    PreventiveActionType.RESOURCE_THROTTLE,
    PreventiveActionType.INTEGRATION_PAUSE,
    PreventiveActionType.CONFIGURATION_UPDATE,
    PreventiveActionType.HUMAN_REVIEW,
    PreventiveActionType.CAPACITY_SCALE,
)


def validate_preventive_action_type(action_type: str) -> PreventiveActionTypeDescriptor:
    """Resolve a qualified or legacy short id against the built-in catalog."""
    stripped = action_type.strip()
    if not stripped:
        raise ValueError("action_type must be non-empty")
    for descriptor in _BUILTIN_TYPES:
        if stripped in {descriptor.qualified_id, f"{descriptor.namespace}.{descriptor.name}"}:
            return descriptor
    raise ValueError(f"unknown preventive action type: {stripped}")


__all__ = [
    "PreventiveActionType",
    "PreventiveActionTypeDescriptor",
    "validate_preventive_action_type",
]
