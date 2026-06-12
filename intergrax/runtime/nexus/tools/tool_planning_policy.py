# © Artur Czarnecki. All rights reserved.

"""Tool planner policy helpers (TOOL-ENG-12)."""

from __future__ import annotations

from typing import Literal

from intergrax.runtime.nexus.config_types import ToolChoiceMode

NativeToolChoice = Literal["auto", "required", "none"]


def tool_choice_for_mode(tools_mode: ToolChoiceMode) -> NativeToolChoice | None:
    if tools_mode == "required":
        return "required"
    if tools_mode == "off":
        return "none"
    return "auto"
