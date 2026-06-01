# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Catalog-backed tool planner (Phase Q+-L.2) — Tier-1 replacement for Tier-0 ToolsAgent in agents."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.runtime.nexus.tools.tool_planner_protocol import ToolPlannerProtocol
from intergrax.tools.registry import ToolRegistry, ToolWiringContext, build_registry_from_profile
from intergrax.tools.registry.profile import ToolProfile
from intergrax.tools.tools_agent import ToolPlanDecision, ToolsAgent


@dataclass
class CatalogToolPlanner:
    """
    Wraps legacy :class:`~intergrax.tools.tools_agent.ToolsAgent` for planning only.

    Tier-2 agents must depend on this type (or another ``ToolPlannerProtocol``), not
    import ``ToolsAgent`` directly.
    """

    _planner: ToolsAgent

    @property
    def llm(self) -> LLMAdapter:
        return self._planner.llm

    @classmethod
    def from_registry(
        cls,
        *,
        llm: LLMAdapter,
        registry: ToolRegistry,
    ) -> CatalogToolPlanner:
        return cls(_planner=ToolsAgent(llm=llm, tools=registry))

    @classmethod
    def from_profile(
        cls,
        *,
        llm: LLMAdapter,
        profile: ToolProfile,
        wiring: Optional[ToolWiringContext] = None,
    ) -> CatalogToolPlanner:
        registry = build_registry_from_profile(profile, wiring_context=wiring)
        return cls.from_registry(llm=llm, registry=registry)

    def plan_tools(
        self,
        input_data: str,
        context: Optional[Any] = None,
        *,
        run_id: str,
    ) -> ToolPlanDecision:
        return self._planner.plan_tools(
            input_data=input_data,
            context=context,
            run_id=run_id,
        )
