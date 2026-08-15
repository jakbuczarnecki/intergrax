# © Artur Czarnecki. All rights reserved.

"""Entry-point registry for custom ``ToolInvocationPattern`` plugins (TOOL-ENG-24)."""

from __future__ import annotations

from intergrax.core.plugins.discovery import (
    EP_TOOL_INVOCATION_PATTERNS,
    get_entry_point_spec,
    instantiate_entry_point_target,
    iter_entry_point_specs,
    load_entry_point_value,
)
from intergrax.runtime.nexus.config_types import ToolInvocationMode
from intergrax.runtime.nexus.tools.tool_invocation_pattern import ToolInvocationPattern


def shipped_pattern_ids() -> frozenset[str]:
    """Stable pattern_id values for all shipped ``ToolInvocationMode`` values."""
    return frozenset(mode.value for mode in ToolInvocationMode)


def load_tool_invocation_pattern(pattern_id: str) -> ToolInvocationPattern | None:
    """Load a pattern by entry-point name from ``intergrax.tool_invocation_patterns``."""
    spec = get_entry_point_spec(EP_TOOL_INVOCATION_PATTERNS, pattern_id)
    if spec is None:
        return None
    loaded = load_entry_point_value(spec.value)
    instance = instantiate_entry_point_target(loaded)
    if not isinstance(instance, ToolInvocationPattern):
        raise TypeError(
            f"Tool invocation entry point {spec.name!r} must return ToolInvocationPattern"
        )
    return instance


def list_tool_invocation_pattern_ids() -> tuple[str, ...]:
    """Return registered entry-point pattern ids (sorted)."""
    return tuple(spec.name for spec in iter_entry_point_specs(EP_TOOL_INVOCATION_PATTERNS))
