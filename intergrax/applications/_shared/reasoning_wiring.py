# © Artur Czarnecki. All rights reserved.

"""Reasoning profile wiring helpers (COG-2.2 / COG-2.3)."""

from __future__ import annotations

from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.prompts.registry.yaml_registry import YamlPromptRegistry
from intergrax.runtime.nexus.planning.engine_plan_models import (
    DEFAULT_PLANNER_FALLBACK_CLARIFY_QUESTION,
    DEFAULT_PLANNER_NEXT_STEP_RULES_PROMPT,
    DEFAULT_PLANNER_REPLAN_SYSTEM_PROMPT,
    DEFAULT_PLANNER_SYSTEM_PROMPT,
    PlannerPromptConfig,
)
from intergrax.runtime.nexus.tools.tool_planning_config import ToolPlanningConfig


def resolve_tool_planning_config(
    env: ApplicationEnvironmentProfile,
    *,
    registry: YamlPromptRegistry | None = None,
    catalog_path: str | None = None,
) -> ToolPlanningConfig:
    """COG-2.2 — bind tool planner prompts from ``ReasoningProfile``."""
    return ToolPlanningConfig.default(
        planner_prompt_id=env.reasoning_profile.tool_planner_prompt_id,
        registry=registry,
        catalog_path=catalog_path,
    )


def resolve_engine_planner_prompt_config(
    env: ApplicationEnvironmentProfile,
    *,
    registry: YamlPromptRegistry | None = None,
    catalog_path: str | None = None,
) -> PlannerPromptConfig:
    """COG-2.3 — registry-backed engine planner prompt config."""
    prompt_id = env.reasoning_profile.engine_planner_prompt_id
    prompt_kwargs = {"registry": registry, "catalog_path": catalog_path}
    return PlannerPromptConfig(
        version=prompt_id,
        system_prompt=DEFAULT_PLANNER_SYSTEM_PROMPT(**prompt_kwargs),
        replan_system_prompt=DEFAULT_PLANNER_REPLAN_SYSTEM_PROMPT(**prompt_kwargs),
        next_step_rules_prompt=DEFAULT_PLANNER_NEXT_STEP_RULES_PROMPT(**prompt_kwargs),
        fallback_clarify_question=DEFAULT_PLANNER_FALLBACK_CLARIFY_QUESTION(**prompt_kwargs),
    )
