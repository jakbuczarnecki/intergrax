# © Artur Czarnecki. All rights reserved.

"""Resolve Nexus orchestration collaborators from a neutral planning spec (Phase ORCH-1, FLOW)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from intergrax.contracts.application_graph_topology import HostGraphTopologySource
from intergrax.contracts.delegation_budget_policy import DelegationBudgetPolicy
from intergrax.contracts.host_orchestration_environment import HostOrchestrationPlanningSpec
from intergrax.contracts.orchestration_enums import MergeStrategy, MultiAgentOrder
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.runtime.execution.application_graph_spec_to_plan import (
    application_graph_spec_to_nexus_plan,
    should_seed_plan_from_graph_spec,
)
from intergrax.runtime.nexus.llm_task_classifier import LlmTaskClassifier
from intergrax.runtime.nexus.orchestration_capabilities import (
    orchestration_capabilities_from_triggers,
)
from intergrax.runtime.nexus.planning.nexus_llm_plan_builder import build_nexus_plan_from_llm
from intergrax.runtime.nexus.planning.nexus_planner_protocol import NexusTaskPlannerProtocol
from intergrax.runtime.nexus.planning.task_planner import NexusPlan, TaskPlanner
from intergrax.runtime.nexus.rules_task_classifier import RulesTaskClassifier
from intergrax.runtime.nexus.task_classifier import ClassifyingTaskClassifier
from intergrax.runtime.nexus.task_classifier_protocol import NexusTaskClassifierProtocol
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.task.task import Task


class OrchestrationWiringError(ValueError):
    """Raised when orchestration profile kinds cannot be resolved."""


class NexusPlannerKind(str, Enum):  # noqa: UP042
    DEFAULT = "default"
    ENGINE = "engine"


class NexusClassifierKind(str, Enum):  # noqa: UP042
    DEFAULT = "default"
    RULES = "rules"
    LLM = "llm"


@dataclass(frozen=True)
class OrchestrationWiringContext:
    """Optional runtime inputs required by specific planner kinds."""

    llm_adapter: LLMAdapter | None = None
    planner_llm_adapter: LLMAdapter | None = None
    planner_parse_retries: int = 0


@dataclass(frozen=True)
class OrchestrationRuntimeSettings:
    """Resolved orchestration knobs passed into ``NexusLoop`` (Phase FLOW)."""

    max_parallel_nodes: int | None
    max_inflight_nodes: int | None
    max_delegation_depth: int | None
    max_run_retries: int
    merge_strategy: MergeStrategy
    multi_agent_order: MultiAgentOrder
    allow_dynamic_replan: bool


def resolve_task_planner(spec: HostOrchestrationPlanningSpec) -> TaskPlanner:
    return TaskPlanner(multi_agent_order=spec.multi_agent_order)


class EngineBackedNexusPlanner:
    """Nexus planner registered under ``planner_kind=engine`` (Phase FLOW-1 / COG-1.1)."""

    def __init__(
        self,
        llm_adapter: LLMAdapter,
        fallback: TaskPlanner,
        *,
        planner_prompt_id: str = "nexus_task_planner",
        planner_parse_retries: int = 0,
    ) -> None:
        self._llm_adapter = llm_adapter
        self._fallback = fallback
        self._planner_prompt_id = planner_prompt_id
        self._planner_parse_retries = planner_parse_retries

    def plan(self, task: Task, registry: AgentRegistry) -> NexusPlan:
        return build_nexus_plan_from_llm(
            task,
            registry,
            self._llm_adapter,
            fallback=self._fallback,
            planner_prompt_id=self._planner_prompt_id,
            planner_parse_retries=self._planner_parse_retries,
        )


class GraphSpecSeedingPlanner:
    """Seeds ``NexusPlan`` from the live graph topology when the task has no plan id."""

    def __init__(
        self,
        inner: NexusTaskPlannerProtocol,
        topology_source: HostGraphTopologySource,
        *,
        coordination_pattern: str | None = None,
        delegation_budget: DelegationBudgetPolicy | None = None,
    ) -> None:
        self._inner = inner
        self._topology_source = topology_source
        self._coordination_pattern = coordination_pattern
        self._delegation_budget = delegation_budget

    def plan(self, task: Task, registry: AgentRegistry) -> NexusPlan:
        graph_spec = self._topology_source.current_graph_topology()
        if graph_spec is None or not graph_spec.nodes:
            return self._inner.plan(task, registry)
        if should_seed_plan_from_graph_spec(task, graph_spec):
            classification = task.classification or ""
            from intergrax.runtime.nexus.orchestration.swarm_policy import (
                annotate_plan_coordination_pattern,
            )

            plan = application_graph_spec_to_nexus_plan(
                graph_spec,
                task,
                classification=classification,
                delegation_budget=self._delegation_budget,
            )
            metadata = dict(plan.plan_metadata)
            metadata["planner_source"] = "graph_spec"
            plan = plan.model_copy(update={"plan_metadata": metadata})
            return annotate_plan_coordination_pattern(
                plan,
                coordination_pattern=self._coordination_pattern,
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
    if normalized == NexusClassifierKind.RULES.value:
        return NexusClassifierKind.RULES
    if normalized == NexusClassifierKind.LLM.value:
        return NexusClassifierKind.LLM
    raise OrchestrationWiringError(f"Unknown classifier_kind: {raw!r}")


def resolve_nexus_task_planner(
    spec: HostOrchestrationPlanningSpec,
) -> NexusTaskPlannerProtocol:
    """Map the planning spec ``planner_kind`` to a concrete planner implementation."""
    kind = _normalize_planner_kind(spec.planner_kind)
    fallback = resolve_task_planner(spec)

    if kind is NexusPlannerKind.ENGINE:
        if spec.llm_adapter is None:
            raise OrchestrationWiringError("planner_kind='engine' requires OrchestrationWiringContext.llm_adapter")
        planner_llm = spec.planner_llm_adapter or spec.llm_adapter
        inner = EngineBackedNexusPlanner(
            planner_llm,
            fallback=fallback,
            planner_prompt_id=spec.planner_prompt_id,
            planner_parse_retries=spec.planner_parse_retries,
        )
    else:
        inner = fallback

    source = spec.graph_topology_source
    graph_topology = source.current_graph_topology() if source is not None else None
    if source is not None and graph_topology is not None and graph_topology.nodes:
        return GraphSpecSeedingPlanner(
            inner=inner,
            topology_source=source,
            coordination_pattern=spec.coordination_pattern,
            delegation_budget=spec.delegation_budget,
        )
    return inner


def resolve_nexus_task_classifier(
    registry: AgentRegistry,
    spec: HostOrchestrationPlanningSpec,
) -> NexusTaskClassifierProtocol:
    """Map the planning spec ``classifier_kind`` to a classifier implementation."""
    kind = _normalize_classifier_kind(spec.classifier_kind)
    source = spec.graph_topology_source
    graph_topology = source.current_graph_topology() if source is not None else None
    orch_triggers = orchestration_capabilities_from_triggers(
        graph_topology.trigger_capabilities if graph_topology is not None else None,
    )
    pipeline_suffix = graph_topology.pipeline_capability_suffix if graph_topology is not None else ".pipeline"
    intent_routes = list(spec.intent_routes)
    if kind is NexusClassifierKind.DEFAULT:
        return ClassifyingTaskClassifier(
            registry,
            orchestration_trigger_capabilities=orch_triggers,
            pipeline_capability_suffix=pipeline_suffix,
        )
    if kind is NexusClassifierKind.RULES:
        return RulesTaskClassifier(
            registry,
            intent_routes=intent_routes,
            orchestration_trigger_capabilities=orch_triggers,
            pipeline_capability_suffix=pipeline_suffix,
        )
    if kind is NexusClassifierKind.LLM:
        if spec.llm_adapter is None:
            raise OrchestrationWiringError("classifier_kind='llm' requires OrchestrationWiringContext.llm_adapter")
        return LlmTaskClassifier(
            registry,
            spec.llm_adapter,
            intent_routes=intent_routes,
            orchestration_trigger_capabilities=orch_triggers,
            pipeline_capability_suffix=pipeline_suffix,
            classifier_prompt_id=spec.classifier_prompt_id,
        )
    raise OrchestrationWiringError(f"Unhandled classifier_kind: {kind.value}")
