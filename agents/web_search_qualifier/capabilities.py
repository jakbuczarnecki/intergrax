# © Artur Czarnecki. All rights reserved.

"""Capabilities for DIAG-FUNCTIONAL-Q3 web search qualification."""

from __future__ import annotations

WEB_SEARCH_QUALIFICATION_CAPABILITY = "local.workspace.web_search_qualification"

CAPABILITIES: tuple[str, ...] = (WEB_SEARCH_QUALIFICATION_CAPABILITY,)

__all__ = ["CAPABILITIES", "WEB_SEARCH_QUALIFICATION_CAPABILITY"]
