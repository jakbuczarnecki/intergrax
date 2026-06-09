# © Artur Czarnecki. All rights reserved.

"""Bridge LLM output to ``NexusPlan`` for ``planner_kind=engine`` (Phase FLOW-1)."""

from __future__ import annotations

import json
import re
from typing import List

from pydantic import BaseModel, ConfigDict, Field

from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.runtime.nexus.planning.task_planner import NexusPlan, PlanStep, TaskPlanner
from intergrax.runtime.nexus.task_classifier import TaskClassification
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.task.task import Task


class _LlmPlanStepPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    agent_id: str
    description: str = ""
    depends_on: List[str] = Field(default_factory=list)


class _LlmNexusPlanPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    steps: List[_LlmPlanStepPayload] = Field(default_factory=list)


def build_nexus_plan_from_llm(
    task: Task,
    registry: AgentRegistry,
    llm_adapter: LLMAdapter,
    *,
    fallback: TaskPlanner,
) -> NexusPlan:
    """
    Ask the configured LLM for a structured multi-step Nexus plan.

    Falls back to the deterministic ``TaskPlanner`` when parsing fails.
    """
    agent_ids = registry.list_routable_agent_ids(production_mode=False)
    if not agent_ids:
        return fallback.plan(task, registry)

    prompt = (
        "You are a Nexus task planner. Return JSON only with shape "
        '{"steps":[{"agent_id":"...","description":"...","depends_on":["step_id"]}]} '
        f"Use only these agent_ids: {agent_ids}. "
        f"Task message: {task.message!r}. "
        f"Capability: {task.context.capability or ''}. "
        f"Classification: {task.classification or ''}."
    )
    response = llm_adapter.generate_messages(
        [ChatMessage(role="user", content=prompt)],
        run_id=task.task_id,
    )
    raw = response.content.strip()
    payload = _extract_json_object(raw)
    if payload is None:
        return fallback.plan(task, registry)

    try:
        parsed = _LlmNexusPlanPayload.model_validate(payload)
    except Exception:
        return fallback.plan(task, registry)

    if not parsed.steps:
        return fallback.plan(task, registry)

    known = set(agent_ids)
    steps: List[PlanStep] = []
    for index, item in enumerate(parsed.steps, start=1):
        if item.agent_id not in known:
            return fallback.plan(task, registry)
        step_id = f"llm_step_{index}"
        steps.append(
            PlanStep(
                step_id=step_id,
                agent_id=item.agent_id,
                capability=task.context.capability,
                description=item.description or f"llm planned step {index}",
                depends_on=list(item.depends_on),
            )
        )

    criteria = ["non_empty_summary"]
    if task.context.capability:
        criteria.append(f"capability:{task.context.capability}")

    return NexusPlan(
        task_id=task.task_id,
        classification=task.classification or TaskClassification.SINGLE_AGENT_DEFAULT.value,
        steps=steps,
        validation_criteria=criteria,
    )


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
