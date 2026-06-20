# © Artur Czarnecki. All rights reserved.

"""Shared Nexus plan build path (COG-1.1 / COG-1.2)."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import List

from pydantic import BaseModel, ConfigDict, Field

from intergrax.contracts.reasoning_failure import ReasoningFailureKind
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


@dataclass(slots=True)
class PlannerBuildDebug:
    """Unified planner diagnostics surface (COG-1.2)."""

    planner_source: str
    used_fallback: bool = False
    failure_kind: ReasoningFailureKind | None = None
    raw_preview: str = ""


def build_planner_build_debug(
    *,
    planner_source: str,
    used_fallback: bool = False,
    failure_kind: ReasoningFailureKind | None = None,
    raw_preview: str = "",
) -> dict[str, str]:
    payload: dict[str, str] = {
        "planner_source": planner_source,
        "used_fallback": str(used_fallback).lower(),
    }
    if failure_kind is not None:
        payload["failure_kind"] = failure_kind.value
    if raw_preview:
        payload["raw_preview"] = raw_preview[:200]
    return payload


def _attempt_llm_plan_parse(
    raw: str,
) -> _LlmNexusPlanPayload | None:
    payload = _extract_json_object(raw)
    if payload is None:
        return None
    try:
        return _LlmNexusPlanPayload.model_validate(payload)
    except Exception:
        return None


def build_nexus_plan_unified(
    task: Task,
    registry: AgentRegistry,
    llm_adapter: LLMAdapter,
    *,
    fallback: TaskPlanner,
    prompt_text: str,
    planner_source: str = "engine",
    planner_parse_retries: int = 0,
) -> tuple[NexusPlan, PlannerBuildDebug]:
    """Parse LLM JSON via shared extractor; annotate metadata for trace."""
    agent_ids = registry.list_routable_agent_ids(production_mode=False)
    if not agent_ids:
        plan = fallback.plan(task, registry)
        return plan, PlannerBuildDebug(planner_source=planner_source, used_fallback=True)

    max_attempts = 1 + max(0, planner_parse_retries)
    raw = ""
    parsed: _LlmNexusPlanPayload | None = None
    from intergrax.runtime.nexus.context.routing_snapshot_sync import sync_routing_for_task_llm_call

    sync_routing_for_task_llm_call(task)
    for _ in range(max_attempts):
        response = llm_adapter.generate_messages(
            [ChatMessage(role="user", content=prompt_text)],
            run_id=task.task_id,
        )
        raw = response.content.strip()
        parsed = _attempt_llm_plan_parse(raw)
        if parsed is not None:
            break

    if parsed is None:
        plan = fallback.plan(task, registry)
        debug = PlannerBuildDebug(
            planner_source=planner_source,
            used_fallback=True,
            failure_kind=ReasoningFailureKind.PLANNER_PARSE_FAILED,
            raw_preview=raw,
        )
        return plan, debug

    if not parsed.steps:
        plan = fallback.plan(task, registry)
        return plan, PlannerBuildDebug(
            planner_source=planner_source,
            used_fallback=True,
            failure_kind=ReasoningFailureKind.PLANNER_FALLBACK,
        )

    known = set(agent_ids)
    steps: List[PlanStep] = []
    for index, item in enumerate(parsed.steps, start=1):
        if item.agent_id not in known:
            plan = fallback.plan(task, registry)
            return plan, PlannerBuildDebug(
                planner_source=planner_source,
                used_fallback=True,
                failure_kind=ReasoningFailureKind.PLANNER_VALIDATION_FAILED,
            )
        steps.append(
            PlanStep(
                step_id=f"llm_step_{index}",
                agent_id=item.agent_id,
                capability=task.context.capability,
                description=item.description or f"llm planned step {index}",
                depends_on=list(item.depends_on),
            )
        )

    criteria = ["non_empty_summary"]
    if task.context.capability:
        criteria.append(f"capability:{task.context.capability}")

    metadata = build_planner_build_debug(planner_source=planner_source)
    plan = NexusPlan(
        task_id=task.task_id,
        classification=task.classification or TaskClassification.SINGLE_AGENT_DEFAULT.value,
        steps=steps,
        validation_criteria=criteria,
        plan_metadata=metadata,
    )
    return plan, PlannerBuildDebug(planner_source=planner_source)
