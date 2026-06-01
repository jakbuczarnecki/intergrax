# © Artur Czarnecki. All rights reserved.

"""Resolve catalog tool planner for Legal runtime (Phase Q+-L.2)."""

from __future__ import annotations

from typing import Optional

from intergrax.runtime.nexus.tools.catalog_tool_planner import CatalogToolPlanner
from intergrax.runtime.nexus.tools.tool_planner_protocol import ToolPlannerProtocol
from legal.config.legal_agent_config import LegalAgentConfig


def resolve_legal_tool_planner(config: LegalAgentConfig) -> Optional[ToolPlannerProtocol]:
    if config.tool_planner is not None:
        return config.tool_planner
    if config.tools_mode == "off" or config.tool_profile is None:
        return None
    return CatalogToolPlanner.from_profile(
        llm=config.llm_adapter,
        profile=config.tool_profile,
        wiring=config.tool_wiring_context,
    )
