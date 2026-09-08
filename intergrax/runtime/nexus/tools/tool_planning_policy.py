# © Artur Czarnecki. All rights reserved.

"""Tool planner policy helpers (TOOL-ENG-12)."""

from __future__ import annotations

from intergrax.llm_adapters.contracts.native_tool_choice import (
    NativeForcedFunctionChoice,
    NativeToolChoice,
    NativeToolChoiceLiteral,
)
from intergrax.runtime.nexus.config_types import ToolChoiceMode
from intergrax.runtime.nexus.tools.native_planner_action_context import (
    NativePlannerProtocolConfig,
)

__all__ = [
    "NativeForcedFunctionChoice",
    "NativeToolChoice",
    "NativeToolChoiceLiteral",
    "native_tool_choice_for_investigation_round",
    "tool_choice_for_mode",
]


def tool_choice_for_mode(tools_mode: ToolChoiceMode) -> NativeToolChoice | None:
    if tools_mode == "required":
        return "required"
    if tools_mode == "off":
        return "none"
    return "auto"


def native_tool_choice_for_investigation_round(
    *,
    protocol_config: NativePlannerProtocolConfig,
    tools_mode: ToolChoiceMode,
) -> NativeToolChoice | None:
    """Provider-neutral tool choice for certified native investigation rounds."""
    base = tool_choice_for_mode(tools_mode)
    if base == "none":
        return "none"
    if base is not None:
        return base
    return "auto"
