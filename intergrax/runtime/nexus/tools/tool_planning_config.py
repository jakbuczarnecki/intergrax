# © Artur Czarnecki. All rights reserved.

"""Typed configuration for catalog tool planning (Phase U-Typ.2)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from intergrax.prompts.registry.yaml_registry import YamlPromptRegistry
from intergrax.runtime.nexus.tools.tool_planning_prompts import (
    planner_prompt,
    system_context_template,
    system_prompt,
)


@dataclass(slots=True)
class ToolPlanningConfig:
    """Planner-only LLM settings and prompt text for :class:`ToolPlanningService`."""

    temperature: Optional[float] = None
    max_answer_tokens: Optional[int] = None
    system_instructions: str = ""
    system_context_template: str = ""
    planner_instructions: str = ""

    @classmethod
    def default(
        cls,
        *,
        planner_prompt_id: str = "tools_agent_planner",
        registry: YamlPromptRegistry | None = None,
        catalog_path: str | None = None,
    ) -> ToolPlanningConfig:
        prompt_kwargs = {"registry": registry, "catalog_path": catalog_path}
        return cls(
            system_instructions=system_prompt(**prompt_kwargs),
            system_context_template=system_context_template(**prompt_kwargs),
            planner_instructions=planner_prompt(
                prompt_id=planner_prompt_id,
                **prompt_kwargs,
            ),
        )
