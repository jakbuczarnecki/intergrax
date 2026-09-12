# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Extensible preventive action type registry (PREVENTIVE R7)."""

from __future__ import annotations

from dataclasses import dataclass, field

from intergrax.contracts.preventive.actions.action_type import (
    PreventiveActionTypeDescriptor,
    validate_preventive_action_type,
)


@dataclass
class PreventiveActionTypeRegistry:
    """Plugin catalog — backward compatible with built-in types."""

    _extra: dict[str, PreventiveActionTypeDescriptor] = field(default_factory=dict)

    def register(self, descriptor: PreventiveActionTypeDescriptor) -> None:
        self._extra[descriptor.qualified_id] = descriptor
        short = f"{descriptor.namespace}.{descriptor.name}"
        self._extra[short] = descriptor

    def resolve(self, action_type: str) -> PreventiveActionTypeDescriptor:
        stripped = action_type.strip()
        if stripped in self._extra:
            return self._extra[stripped]
        return validate_preventive_action_type(stripped)


__all__ = ["PreventiveActionTypeRegistry"]
