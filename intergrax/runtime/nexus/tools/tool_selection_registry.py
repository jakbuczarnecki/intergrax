# © Artur Czarnecki. All rights reserved.

"""Entry-point registry for custom ``ToolSelectionStrategy`` plugins (TOOL-ENG-26)."""

from __future__ import annotations

from importlib.metadata import entry_points

from intergrax.runtime.nexus.tools.tool_selection import ToolSelectionStrategy

_ENTRY_POINT_GROUP = "intergrax.tool_selection_strategies"


def load_tool_selection_strategy(strategy_id: str) -> ToolSelectionStrategy | None:
    """Load a strategy by entry-point name from ``intergrax.tool_selection_strategies``."""
    try:
        eps = entry_points(group=_ENTRY_POINT_GROUP)
    except TypeError:  # pragma: no cover — Python 3.11
        eps = entry_points().select(group=_ENTRY_POINT_GROUP)

    for ep in eps:
        if ep.name != strategy_id:
            continue
        loaded = ep.load()
        if isinstance(loaded, type):
            instance: ToolSelectionStrategy = loaded()
        else:
            instance = loaded
        return instance
    return None


def list_tool_selection_strategy_ids() -> tuple[str, ...]:
    """Return registered entry-point strategy ids (sorted)."""
    try:
        eps = entry_points(group=_ENTRY_POINT_GROUP)
    except TypeError:  # pragma: no cover — Python 3.11
        eps = entry_points().select(group=_ENTRY_POINT_GROUP)
    return tuple(sorted(ep.name for ep in eps))
