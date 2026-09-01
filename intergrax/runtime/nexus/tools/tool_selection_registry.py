# © Artur Czarnecki. All rights reserved.

"""Entry-point registry for custom ``ToolSelectionStrategy`` plugins (TOOL-ENG-26)."""

from __future__ import annotations

from intergrax.core.plugins.discovery import (
    EP_TOOL_SELECTION_STRATEGIES,
    get_entry_point_spec,
    instantiate_entry_point_target,
    iter_entry_point_specs,
    load_entry_point_value,
)
from intergrax.runtime.nexus.tools.tool_selection import ToolSelectionStrategy

_ENTRY_POINT_GROUP = EP_TOOL_SELECTION_STRATEGIES


def load_tool_selection_strategy(strategy_id: str) -> ToolSelectionStrategy | None:
    """Load a strategy by entry-point name from ``intergrax.tool_selection_strategies``."""
    spec = get_entry_point_spec(_ENTRY_POINT_GROUP, strategy_id)
    if spec is None:
        return None
    loaded = load_entry_point_value(spec.value)
    instance = instantiate_entry_point_target(loaded)
    return instance


def list_tool_selection_strategy_ids() -> tuple[str, ...]:
    """Return registered entry-point strategy ids (sorted)."""
    return tuple(spec.name for spec in iter_entry_point_specs(_ENTRY_POINT_GROUP))
