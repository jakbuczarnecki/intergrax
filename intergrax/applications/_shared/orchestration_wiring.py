# © Artur Czarnecki. All rights reserved.

"""Resolve Nexus orchestration collaborators from ``ApplicationEnvironmentProfile`` (Phase ORCH-1)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from intergrax.applications._shared.graph_spec_to_plan import (
    application_graph_spec_to_nexus_plan,
    should_seed_plan_from_graph_spec,
)
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications.contracts.graph_spec import ApplicationGraphSpec
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.runtime.nexus.planning.nexus_planner_protocol import NexusTaskPlannerProtocol
from intergrax.runtime.nexus.planning.task_planner import NexusPlan, TaskPlanner
from intergrax.runtime.nexus.task_classifier import ClassifyingTaskClassifier
from intergrax.runtime.nexus.task_classifier_protocol import NexusTaskClassifierProtocol
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.task.task import Task


class OrchestrationWiringError(ValueError):
    """Raised when orchestration profile kinds cannot be resolved."""


class NexusPlannerKind(str, Enum):
    DEFAULT = "default"
    ENGINE = "engine"


class NexusClassifierKind(str, Enum):
    DEFAULT = "default"


@dataclass(frozen=True)
class OrchestrationWiringContext:
    """Optional runtime inputs required by specific planner kinds."""

    llm_adapter: LLMAdapter | None = None


class EngineBackedNexusPlanner:
    """
    Nexus planner registered under ``planner_kind=engine``.

    Validates LLM availability at bootstrap; uses deterministic ``TaskPlanner`` for
    Nexus-level steps until async ``EnginePlan`` bridging is product-scheduled.
    """

    def __init__(self, llm_adapter: LLMAdapter) -> None:
        self._llm_adapter = llm_adapter
        self._inner = TaskPlanner()

    def plan(self, task: Task, registry: AgentRegistry) -> NexusPlan:
        _ = self._llm_adapter
        return self._inner.plan(task, registry)


class GraphSpecSeedingPlanner:
    """Seeds ``NexusPlan`` from ``ApplicationGraphSpec`` when the task has no plan id."""

    def __init__(
        self,
        inner: NexusTaskPlannerProtocol,
        graph_spec: ApplicationGraphSpec,
    ) -> None:
        self._inner = inner
        self._graph_spec = graph_spec

    def plan(self, task: Task, registry: AgentRegistry) -> NexusPlan:
        if should_seed_plan_from_graph_spec(task) and self._graph_spec.nodes:
            classification = task.classification or ""
            return application_graph_spec_to_nexus_plan(
                self._graph_spec,
                task,
                classification=classification,
            )
        return self._inner.plan(task, registry)


def _normalize_planner_kind(raw: str | None) -> NexusPlannerKind:
    if raw is None or not raw.strip():
        return NexusPlannerKind.DEFAULT
    normalized = raw.strip().lower()
    if normalized == NexusPlannerKind.DEFAULT.value:
        return NexusPlannerKind.DEFAULT
    if normalized == NexusPlannerKind.ENGINE.value:
        return NexusPlannerKind.ENGINE
    raise OrchestrationWiringError(f"Unknown planner_kind: {raw!r}")


def _normalize_classifier_kind(raw: str | None) -> NexusClassifierKind:
    if raw is None or not raw.strip():
        return NexusClassifierKind.DEFAULT
    normalized = raw.strip().lower()
    if normalized == NexusClassifierKind.DEFAULT.value:
        return NexusClassifierKind.DEFAULT
    raise OrchestrationWiringError(f"Unknown classifier_kind: {raw!r}")


def resolve_nexus_task_planner(
    env: ApplicationEnvironmentProfile,
    *,
    wiring_context: OrchestrationWiringContext | None = None,
) -> NexusTaskPlannerProtocol:
    """Map ``OrchestrationProfile.planner_kind`` to a concrete planner implementation."""
    kind = _normalize_planner_kind(env.orchestration_profile.planner_kind)
    context = wiring_context or OrchestrationWiringContext()

    if kind is NexusPlannerKind.ENGINE:
        if context.llm_adapter is None:
            raise OrchestrationWiringError(
                "planner_kind='engine' requires OrchestrationWiringContext.llm_adapter"
            )
        inner: NexusTaskPlannerProtocol = EngineBackedNexusPlanner(context.llm_adapter)
    else:
        inner = TaskPlanner()

    graph_spec = env.graph_spec
    if graph_spec is not None and graph_spec.nodes:
        return GraphSpecSeedingPlanner(inner=inner, graph_spec=graph_spec)
    return inner


def resolve_nexus_task_classifier(
    registry: AgentRegistry,
    env: ApplicationEnvironmentProfile,
) -> NexusTaskClassifierProtocol:
    """Map ``OrchestrationProfile.classifier_kind`` to a classifier implementation."""
    kind = _normalize_classifier_kind(env.orchestration_profile.classifier_kind)
    if kind is NexusClassifierKind.DEFAULT:
        return ClassifyingTaskClassifier(registry)
    raise OrchestrationWiringError(f"Unhandled classifier_kind: {kind.value}")


def resolve_max_parallel_nodes(env: ApplicationEnvironmentProfile) -> int | None:
    """Return graph batch concurrency cap from the environment profile."""
    return env.orchestration_profile.max_parallel_nodes
