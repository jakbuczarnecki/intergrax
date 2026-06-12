# © Artur Czarnecki. All rights reserved.

"""Entry-point registry for custom ``ToolInvocationPattern`` plugins (TOOL-ENG-24)."""

from __future__ import annotations

from importlib.metadata import entry_points

from intergrax.runtime.nexus.config_types import ToolInvocationMode
from intergrax.runtime.nexus.tools.tool_invocation_pattern import ToolInvocationPattern

_ENTRY_POINT_GROUP = "intergrax.tool_invocation_patterns"


def shipped_pattern_ids() -> frozenset[str]:
    """Stable pattern_id values for all shipped ``ToolInvocationMode`` values."""
    return frozenset(mode.value for mode in ToolInvocationMode)


def load_tool_invocation_pattern(pattern_id: str) -> ToolInvocationPattern | None:
    """Load a pattern by entry-point name from ``intergrax.tool_invocation_patterns``."""
    try:
        eps = entry_points(group=_ENTRY_POINT_GROUP)
    except TypeError:  # pragma: no cover — Python 3.11
        eps = entry_points().select(group=_ENTRY_POINT_GROUP)

    for ep in eps:
        if ep.name != pattern_id:
            continue
        loaded = ep.load()
        if isinstance(loaded, type):
            instance: ToolInvocationPattern = loaded()
        else:
            instance = loaded
        return instance
    return None


def list_tool_invocation_pattern_ids() -> tuple[str, ...]:
    """Return registered entry-point pattern ids (sorted)."""
    try:
        eps = entry_points(group=_ENTRY_POINT_GROUP)
    except TypeError:  # pragma: no cover — Python 3.11
        eps = entry_points().select(group=_ENTRY_POINT_GROUP)
    return tuple(sorted(ep.name for ep in eps))
