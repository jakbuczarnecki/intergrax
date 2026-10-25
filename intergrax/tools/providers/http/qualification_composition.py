# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Isolated TOOL-LIFE-X HTTP Class B qualification registration (not production HTTP bundle)."""

from __future__ import annotations

from intergrax.tools.providers.http.idempotent_post_qual import (
    HTTP_IDEMPOTENT_POST_QUAL_TOOL_ID,
    register_http_idempotent_post_qual_tool,
)
from intergrax.tools.registry.runtime import ToolRegistry
from intergrax.tools.registry.wiring import ToolWiringContext

TOOL_LIFE_HTTP_CLASS_B_QUALIFICATION_TOOL_IDS: tuple[str, ...] = (
    HTTP_IDEMPOTENT_POST_QUAL_TOOL_ID,
)


def register_tool_life_http_class_b_qualification_tools(
    registry: ToolRegistry,
    ctx: ToolWiringContext,
) -> None:
    """Register sanctioned Class B HTTP qual tools — explicit qualification composition only."""
    register_http_idempotent_post_qual_tool(registry, ctx)


__all__ = [
    "HTTP_IDEMPOTENT_POST_QUAL_TOOL_ID",
    "TOOL_LIFE_HTTP_CLASS_B_QUALIFICATION_TOOL_IDS",
    "register_tool_life_http_class_b_qualification_tools",
]
