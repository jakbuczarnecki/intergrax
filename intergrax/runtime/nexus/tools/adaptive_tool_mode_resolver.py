# © Artur Czarnecki. All rights reserved.

"""Per-run adaptive tool selection and invocation mode resolution (TOOL-ENG-10)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.runtime.nexus.config_types import ToolInvocationMode, ToolSelectionMode
from intergrax.tools.registry import ToolRegistry


@dataclass(frozen=True, slots=True)
class AdaptiveToolModeRecommendation:
    tool_selection_mode: ToolSelectionMode
    tool_invocation_mode: ToolInvocationMode | None = None


def recommend_tool_modes(
    *,
    registry: ToolRegistry,
    query: str,
) -> AdaptiveToolModeRecommendation:
    """
    Rule-based L6 / 2a mode pick from catalog scale (AHI routing hook).

    Large catalogs → semantic narrowing; medium → hierarchical or keyword; small → full catalog.
    """
    _ = query
    tool_count = len(list(registry.list()))
    if tool_count > 80:
        return AdaptiveToolModeRecommendation(
            tool_selection_mode=ToolSelectionMode.SEMANTIC,
            tool_invocation_mode=ToolInvocationMode.SINGLE_PASS,
        )
    if tool_count > 30:
        return AdaptiveToolModeRecommendation(
            tool_selection_mode=ToolSelectionMode.HIERARCHICAL,
            tool_invocation_mode=ToolInvocationMode.SINGLE_PASS,
        )
    if tool_count > 15:
        return AdaptiveToolModeRecommendation(
            tool_selection_mode=ToolSelectionMode.RETRIEVAL_TOP_K,
            tool_invocation_mode=ToolInvocationMode.SINGLE_PASS,
        )
    return AdaptiveToolModeRecommendation(
        tool_selection_mode=ToolSelectionMode.FULL_CATALOG,
        tool_invocation_mode=ToolInvocationMode.SINGLE_PASS,
    )
