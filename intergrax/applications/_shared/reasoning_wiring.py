# © Artur Czarnecki. All rights reserved.

"""Reasoning profile wiring helpers (COG-2.2 / COG-2.3)."""

from __future__ import annotations

from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.prompts.registry.yaml_registry import YamlPromptRegistry
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


def resolve_replan_policy_context(
    env: ApplicationEnvironmentProfile,
) -> dict[str, bool]:
    """AUDIT-IDEAL-7.2 — policy context for Nexus dynamic replan."""
    if not env.orchestration_profile.allow_dynamic_replan:
        return {}
    return {"nexus_replan_boundary": True}
