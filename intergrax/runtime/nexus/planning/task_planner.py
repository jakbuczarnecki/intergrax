# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from typing import List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field

from intergrax.runtime.nexus.task_classifier import TaskClassification
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.task.task import Task


class PlanStep(BaseModel):
    """Single step in a Nexus task plan (§10.3)."""

    step_id: str
    agent_id: Optional[str] = None
    capability: Optional[str] = None
    description: str = ""
    depends_on: List[str] = Field(default_factory=list)


class NexusPlan(BaseModel):
    """Minimal task-level plan produced before agent execution."""

    plan_id: str = Field(default_factory=lambda: f"plan_{uuid4().hex}")
    task_id: str
    classification: str
    steps: List[PlanStep] = Field(default_factory=list)
    validation_criteria: List[str] = Field(default_factory=list)


class TaskPlanner:
    """
    Minimal Nexus planner (Phase B.2).

    Produces a sequential plan; multi-step plans prepare for ExecutionGraph (Phase C).
    """

    def plan(self, task: Task, registry: AgentRegistry) -> NexusPlan:
        classification = task.classification or TaskClassification.SINGLE_AGENT_DEFAULT.value

        if classification == TaskClassification.UNSUPPORTED.value:
            return NexusPlan(
                task_id=task.task_id,
                classification=classification,
                steps=[],
            )

        if classification == TaskClassification.MULTI_AGENT.value:
            return self._multi_agent_plan(task, registry, classification)

        if self._is_research_pipeline(task):
            return self._research_pipeline_plan(task, registry, classification)

        agent_id = task.agent_id
        capability = task.context.capability

        if not agent_id and capability:
            matches = registry.find_by_capability(capability)
            if matches:
                agent_id = matches[0].get_contract().id

        if not agent_id:
            ids = registry.list_agent_ids()
            agent_id = ids[0] if ids else None

        step = PlanStep(
            step_id="step_1",
            agent_id=agent_id,
            capability=capability,
            description="execute primary agent",
        )

        criteria = ["non_empty_summary"]
        if capability:
            criteria.append(f"capability:{capability}")

        return NexusPlan(
            task_id=task.task_id,
            classification=classification,
            steps=[step],
            validation_criteria=criteria,
        )

    def _multi_agent_plan(
        self,
        task: Task,
        registry: AgentRegistry,
        classification: str,
    ) -> NexusPlan:
        capability = task.context.capability or ""
        agents = registry.find_by_capability(capability) if capability else []
        steps: List[PlanStep] = []
        prev_id: Optional[str] = None
        for idx, agent in enumerate(agents, start=1):
            step_id = f"step_{idx}"
            steps.append(
                PlanStep(
                    step_id=step_id,
                    agent_id=agent.get_contract().id,
                    capability=capability,
                    description=f"multi-agent step {idx}",
                    depends_on=[prev_id] if prev_id else [],
                )
            )
            prev_id = step_id
        return NexusPlan(
            task_id=task.task_id,
            classification=classification,
            steps=steps,
            validation_criteria=["non_empty_summary"],
        )

    @staticmethod
    def _is_research_pipeline(task: Task) -> bool:
        cap = task.context.capability or ""
        intent = task.context.intent or ""
        return cap == "research.pipeline" or intent == "research_summarize"

    def _research_pipeline_plan(
        self,
        task: Task,
        registry: AgentRegistry,
        classification: str,
    ) -> NexusPlan:
        research_id = self._agent_for_capability(registry, "research.web_search", "research")
        summary_id = self._agent_for_capability(registry, "research.summarize", "research-summary")

        return NexusPlan(
            task_id=task.task_id,
            classification=classification or TaskClassification.MULTI_AGENT.value,
            steps=[
                PlanStep(
                    step_id="research",
                    agent_id=research_id,
                    capability="research.web_search",
                    description="gather research findings (stub)",
                ),
                PlanStep(
                    step_id="summarize",
                    agent_id=summary_id,
                    capability="research.summarize",
                    description="summarize research findings",
                    depends_on=["research"],
                ),
            ],
            validation_criteria=["non_empty_summary", "capability:research.pipeline"],
        )

    @staticmethod
    def _agent_for_capability(
        registry: AgentRegistry,
        capability: str,
        fallback_id: str,
    ) -> str:
        matches = registry.find_by_capability(capability)
        if matches:
            return matches[0].get_contract().id
        if registry.has(fallback_id):
            return fallback_id
        ids = registry.list_agent_ids()
        return ids[0] if ids else fallback_id
