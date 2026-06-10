# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Catalog-backed tool planner (Phase Q+-L.2, T-Ops.5)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Optional, Union

from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.prompts.registry.yaml_registry import YamlPromptRegistry
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

    Uses :class:`~intergrax.runtime.nexus.tools.tool_planning_service.ToolPlanningService`.
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
        prompt_registry: YamlPromptRegistry | None = None,
        prompt_catalog_path: str | None = None,
        planner_prompt_id: str = "tools_agent_planner",
    ) -> CatalogToolPlanner:
        from intergrax.runtime.nexus.tools.tool_planning_config import ToolPlanningConfig

        config = ToolPlanningConfig.default(
            planner_prompt_id=planner_prompt_id,
            registry=prompt_registry,
            catalog_path=prompt_catalog_path,
        )
        return cls(_service=ToolPlanningService(llm=llm, tools=registry, config=config))

    @classmethod
    def from_profile(
        cls,
        *,
        llm: LLMAdapter,
        profile: ToolProfile,
        wiring: Optional[ToolWiringContext] = None,
        prompt_registry: YamlPromptRegistry | None = None,
        prompt_catalog_path: str | None = None,
    ) -> CatalogToolPlanner:
        registry = build_registry_from_profile(profile, wiring_context=wiring)
        return cls.from_registry(
            llm=llm,
            registry=registry,
            prompt_registry=prompt_registry,
            prompt_catalog_path=prompt_catalog_path,
        )

    def plan_tools(
        self,
        input_data: Union[str, list[ChatMessage]],
        context: Optional[Any] = None,
        *,
        run_id: str,
        allowed_tool_ids: Sequence[str] | None = None,
    ) -> ToolPlanDecision:
        return self._service.plan_tools(
            input_data=input_data,
            context=context,
            run_id=run_id,
            allowed_tool_ids=allowed_tool_ids,
        )
