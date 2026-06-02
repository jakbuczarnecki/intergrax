# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Catalog-backed tool planner (Phase Q+-L.2, T-Ops.5) — no ``ToolsAgent`` wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.runtime.nexus.tools.tool_planner_protocol import ToolPlannerProtocol
from intergrax.runtime.nexus.tools.tool_planner_trackable import ToolPlannerTrackable
from intergrax.runtime.nexus.tools.tool_planning_service import ToolPlanningService
from intergrax.tools.registry import ToolRegistry, ToolWiringContext, build_registry_from_profile
from intergrax.tools.registry.profile import ToolProfile
from intergrax.tools.core.tool_plan_decision import ToolPlanDecision


@dataclass
class CatalogToolPlanner(ToolPlannerTrackable):
    """
    Tier-1 catalog planner implementing :class:`ToolPlannerProtocol`.

    Uses :class:`~intergrax.runtime.nexus.tools.tool_planning_service.ToolPlanningService`
    instead of legacy :class:`~intergrax.tools.tools_agent.ToolsAgent`.
    """

    _service: ToolPlanningService

    @property
    def llm(self) -> LLMAdapter:
        return self._service.llm

    @classmethod
    def from_registry(
        cls,
        *,
        llm: LLMAdapter,
        registry: ToolRegistry,
    ) -> CatalogToolPlanner:
        return cls(_service=ToolPlanningService(llm=llm, tools=registry))

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
        return self._service.plan_tools(
            input_data=input_data,
            context=context,
            run_id=run_id,
        )
