# © Artur Czarnecki. All rights reserved.

"""Entry-point registry bridge for custom invocation patterns (TOOL-ENG-24)."""

from __future__ import annotations

from intergrax.runtime.nexus.config_types import ToolInvocationMode
from intergrax.runtime.nexus.tools.public_tool_invocation_pattern_bridge import (
    bridge_public_tool_invocation_pattern,
)
from intergrax.runtime.nexus.tools.tool_invocation_pattern import NexusToolInvocationPattern
from intergrax.tools.invocation_pattern.registry import (
    list_tool_invocation_pattern_ids as _list_public_pattern_ids,
)
from intergrax.tools.invocation_pattern.registry import (
    load_tool_invocation_pattern as _load_public_pattern,
)


def shipped_pattern_ids() -> frozenset[str]:
    """Stable pattern_id values for all shipped ``ToolInvocationMode`` values."""
    return frozenset(mode.value for mode in ToolInvocationMode)


def load_tool_invocation_pattern(pattern_id: str) -> NexusToolInvocationPattern | None:
    """Load a public EP pattern and adapt it for Nexus execution."""
    public = _load_public_pattern(pattern_id)
    if public is None:
        return None
    return bridge_public_tool_invocation_pattern(public)


def list_tool_invocation_pattern_ids() -> tuple[str, ...]:
    """Return registered entry-point pattern ids (sorted)."""
    return _list_public_pattern_ids()
