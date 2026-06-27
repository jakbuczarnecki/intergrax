# © Artur Czarnecki. All rights reserved.

"""Resolve Nexus orchestration collaborators from ``ApplicationEnvironmentProfile`` (Phase ORCH-1, FLOW)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from intergrax.applications._shared.delegation_budget_wiring import (
    DelegationBudgetPolicy,
    resolve_delegation_budget_policy,
)
from intergrax.applications._shared.graph_spec_to_plan import (
    application_graph_spec_to_nexus_plan,
    should_seed_plan_from_graph_spec,
)
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications.contracts.graph_spec import ApplicationGraphSpec
from intergrax.contracts.orchestration_enums import MergeStrategy, MultiAgentOrder
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.runtime.nexus.orchestration_capabilities import (
    orchestration_capabilities_from_triggers,
)
from intergrax.runtime.nexus.planning.nexus_llm_plan_builder import build_nexus_plan_from_llm
from intergrax.runtime.nexus.planning.nexus_planner_protocol import NexusTaskPlannerProtocol
from intergrax.runtime.nexus.planning.task_planner import NexusPlan, TaskPlanner
from intergrax.runtime.nexus.llm_task_classifier import LlmTaskClassifier
from intergrax.runtime.nexus.rules_task_classifier import RulesTaskClassifier
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


def _resolve_multi_agent_order(raw: str) -> MultiAgentOrder:
    try:
        return MultiAgentOrder(raw)
    except ValueError:
        return MultiAgentOrder.REGISTRY


def _resolve_merge_strategy(raw: str) -> MergeStrategy:
    try:
        return MergeStrategy(raw)
    except ValueError:
        return MergeStrategy.CONCAT


def resolve_task_planner(env: ApplicationEnvironmentProfile) -> TaskPlanner:
    order = _resolve_multi_agent_order(env.orchestration_profile.multi_agent_order)
    return TaskPlanner(multi_agent_order=order)


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
    """Seeds ``NexusPlan`` from ``ApplicationGraphSpec`` when the task has no plan id."""

    def __init__(
        self,
        inner: NexusTaskPlannerProtocol,
        graph_spec: ApplicationGraphSpec,
        *,
        coordination_pattern: str | None = None,
        delegation_budget: DelegationBudgetPolicy | None = None,
    ) -> None:
        self._inner = inner
        self._graph_spec = graph_spec
        self._coordination_pattern = coordination_pattern
        self._delegation_budget = delegation_budget

    def plan(self, task: Task, registry: AgentRegistry) -> NexusPlan:
        if should_seed_plan_from_graph_spec(task, self._graph_spec):
            classification = task.classification or ""
            from intergrax.runtime.nexus.orchestration.swarm_policy import (
                annotate_plan_coordination_pattern,
            )

            plan = application_graph_spec_to_nexus_plan(
                self._graph_spec,
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
    env: ApplicationEnvironmentProfile,
    *,
    wiring_context: OrchestrationWiringContext | None = None,
) -> NexusTaskPlannerProtocol:
    """Map ``OrchestrationProfile.planner_kind`` to a concrete planner implementation."""
    kind = _normalize_planner_kind(env.orchestration_profile.planner_kind)
    context = wiring_context or OrchestrationWiringContext()
    fallback = resolve_task_planner(env)

    if kind is NexusPlannerKind.ENGINE:
        if context.llm_adapter is None:
            raise OrchestrationWiringError(
                "planner_kind='engine' requires OrchestrationWiringContext.llm_adapter"
            )
        planner_llm = context.planner_llm_adapter or context.llm_adapter
        inner = EngineBackedNexusPlanner(
            planner_llm,
            fallback=fallback,
            planner_prompt_id=env.reasoning_profile.planner_prompt_id,
            planner_parse_retries=env.reasoning_profile.planner_parse_retries,
        )
    else:
        inner = fallback

    graph_spec = env.graph_spec
    if graph_spec is not None and graph_spec.nodes:
        return GraphSpecSeedingPlanner(
            inner=inner,
            graph_spec=graph_spec,
            coordination_pattern=env.orchestration_profile.coordination_pattern,
            delegation_budget=resolve_delegation_budget_policy(env),
        )
    return inner


def resolve_nexus_task_classifier(
    registry: AgentRegistry,
    env: ApplicationEnvironmentProfile,
    *,
    wiring_context: OrchestrationWiringContext | None = None,
) -> NexusTaskClassifierProtocol:
    """Map ``OrchestrationProfile.classifier_kind`` to a classifier implementation."""
    kind = _normalize_classifier_kind(env.orchestration_profile.classifier_kind)
    context = wiring_context or OrchestrationWiringContext()
    orch_triggers = orchestration_capabilities_from_triggers(
        env.graph_spec.trigger_capabilities if env.graph_spec is not None else None,
    )
    pipeline_suffix = (
        env.graph_spec.pipeline_capability_suffix
        if env.graph_spec is not None
        else ".pipeline"
    )
    intent_routes = list(env.orchestration_profile.intent_routes)
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
        if context.llm_adapter is None:
            raise OrchestrationWiringError(
                "classifier_kind='llm' requires OrchestrationWiringContext.llm_adapter"
            )
        return LlmTaskClassifier(
            registry,
            context.llm_adapter,
            intent_routes=intent_routes,
            orchestration_trigger_capabilities=orch_triggers,
            pipeline_capability_suffix=pipeline_suffix,
            classifier_prompt_id=env.reasoning_profile.classifier_prompt_id,
        )
    raise OrchestrationWiringError(f"Unhandled classifier_kind: {kind.value}")


def resolve_orchestration_runtime_settings(
    env: ApplicationEnvironmentProfile,
) -> OrchestrationRuntimeSettings:
    profile = env.orchestration_profile
    return OrchestrationRuntimeSettings(
        max_parallel_nodes=profile.max_parallel_nodes,
        max_inflight_nodes=profile.max_inflight_nodes,
        max_delegation_depth=profile.max_delegation_depth,
        max_run_retries=profile.max_run_retries,
        merge_strategy=_resolve_merge_strategy(profile.merge_strategy),
        multi_agent_order=_resolve_multi_agent_order(profile.multi_agent_order),
        allow_dynamic_replan=profile.allow_dynamic_replan,
    )


def resolve_max_parallel_nodes(env: ApplicationEnvironmentProfile) -> int | None:
    """Return graph batch concurrency cap from the environment profile."""
    return env.orchestration_profile.max_parallel_nodes


def resolve_max_inflight_nodes(env: ApplicationEnvironmentProfile) -> int | None:
    """Return graph backpressure cap from the environment profile."""
    return env.orchestration_profile.max_inflight_nodes
