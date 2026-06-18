# © Artur Czarnecki. All rights reserved.

"""Reasoning profile wiring helpers (COG-2.2 / COG-2.3 / COG-PROD)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
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


def resolve_reasoning_task_metadata(
    env: ApplicationEnvironmentProfile,
) -> dict[str, object]:
    """Plane 2 reasoning metadata injected on Nexus tasks (COG-LC-S2)."""
    metadata: dict[str, object] = {
        "engine_planner_prompt_id": resolve_engine_planner_prompt_config(env).prompt_id,
    }
    replan = resolve_replan_policy_context(env)
    if replan:
        metadata["replan_policy.v1"] = replan
    return metadata


def resolve_planner_llm_adapter(
    env: ApplicationEnvironmentProfile,
    *,
    producer_adapter: LLMAdapter,
) -> LLMAdapter:
    """
    Resolve planner LLM with producer/planner separation (COG-PROD.1).

    Precedence:
    1. ``ReasoningProfile.planner_llm_profile`` when set
    2. Producer adapter
    """
    separate = env.reasoning_profile.planner_llm_profile
    if separate is not None:
        return separate.create_adapter()
    return producer_adapter


def resolve_planner_model_id(env: ApplicationEnvironmentProfile) -> str | None:
    """Observability and policy deny-list key for the planning-phase LLM."""
    reasoning = env.reasoning_profile
    if reasoning.planner_llm_profile_id:
        return reasoning.planner_llm_profile_id
    if reasoning.planner_llm_profile is not None:
        model = reasoning.planner_llm_profile.model
        return model if model else None
    if env.llm_profile is not None and env.llm_profile.model:
        return env.llm_profile.model
    return None


@dataclass(slots=True)
class EnginePlannerPromptConfig:
    """Registry binding for agent-level engine step planners (COG-2.3 / COG-PROD.3)."""

    prompt_id: str


def resolve_engine_planner_prompt_config(
    env: ApplicationEnvironmentProfile,
) -> EnginePlannerPromptConfig:
    """COG-2.3 — bind ``ReasoningProfile.engine_planner_prompt_id`` for agent cognition."""
    return EnginePlannerPromptConfig(prompt_id=env.reasoning_profile.engine_planner_prompt_id)
