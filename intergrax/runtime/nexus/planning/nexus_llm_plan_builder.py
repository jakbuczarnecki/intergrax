# © Artur Czarnecki. All rights reserved.

"""Bridge LLM output to ``NexusPlan`` for ``planner_kind=engine`` (Phase FLOW-1 / COG-1.*)."""

from __future__ import annotations

import json
import re

from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.runtime.nexus.planning.nexus_plan_bridge import (
    build_nexus_plan_unified,
    build_planner_build_debug,
)
from intergrax.runtime.nexus.planning.nexus_planner_prompts import nexus_task_planner_prompt
from intergrax.runtime.nexus.planning.task_planner import NexusPlan, TaskPlanner
from intergrax.runtime.registry.agent_registry_read import AgentRegistryRead
from intergrax.runtime.task.task import Task


def build_nexus_plan_from_llm(
    task: Task,
    registry: AgentRegistryRead,
    llm_adapter: LLMAdapter,
    *,
    fallback: TaskPlanner,
    planner_prompt_id: str = "nexus_task_planner",
    planner_parse_retries: int = 0,
) -> NexusPlan:
    """Ask the configured LLM for a structured multi-step Nexus plan."""
    agent_ids = registry.list_routable_agent_ids(production_mode=False)
    prompt = nexus_task_planner_prompt(
        prompt_id=planner_prompt_id,
        agent_ids=list(agent_ids),
        task_message=task.message,
        capability=task.context.capability or "",
        classification=task.classification or "",
    )
    plan, debug = build_nexus_plan_unified(
        task,
        registry,
        llm_adapter,
        fallback=fallback,
        prompt_text=prompt,
        planner_source="engine",
        planner_parse_retries=planner_parse_retries,
    )
    metadata = dict(plan.plan_metadata)
    metadata.update(
        build_planner_build_debug(
            planner_source=debug.planner_source,
            used_fallback=debug.used_fallback,
            failure_kind=debug.failure_kind,
            raw_preview=debug.raw_preview,
        )
    )
    return plan.model_copy(update={"plan_metadata": metadata})


def _extract_json_object(raw: str) -> dict[str, object] | None:
    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if match is None:
        return None
    try:
        loaded = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if isinstance(loaded, dict):
        return loaded
    return None
