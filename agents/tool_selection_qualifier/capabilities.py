# © Artur Czarnecki. All rights reserved.

"""Capabilities for DIAG-FUNCTIONAL-Q2 tool-selection qualification."""

from __future__ import annotations

TOOL_SELECTION_QUALIFICATION_CAPABILITY = "local.workspace.tool_selection_qualification"

CAPABILITIES: tuple[str, ...] = (TOOL_SELECTION_QUALIFICATION_CAPABILITY,)

__all__ = ["CAPABILITIES", "TOOL_SELECTION_QUALIFICATION_CAPABILITY"]
