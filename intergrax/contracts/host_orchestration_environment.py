# © Artur Czarnecki. All rights reserved.

"""Neutral resolved host-orchestration environment values (Tier-3 builds, Tier-1 applies).

All fields are frozen, named and Tier-0 typed so Tier-1 never reads the Tier-3 profile
models and no field can be silently dropped by a profile round-trip.
"""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.contracts.application_graph_topology import HostGraphTopologySource
from intergrax.contracts.attempt_lifecycle import AttemptLifecyclePersistenceProvider
from intergrax.contracts.context_assembly import TaskContextAssemblyOptions
from intergrax.contracts.context_budget import ContextBudgetPolicy
from intergrax.contracts.delegation_budget_policy import DelegationBudgetPolicy
from intergrax.contracts.execution_lineage import ExecutionLineagePersistenceProvider
from intergrax.contracts.execution_mode import ExecutionMode
from intergrax.contracts.execution_terminal import ExecutionTerminalPersistenceProvider
from intergrax.contracts.intent_route import IntentRoute
from intergrax.contracts.orchestration_enums import MergeStrategy, MultiAgentOrder
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter


@dataclass(frozen=True, slots=True)
class HostContextCompositionSpec:
    """Context engine and manager inputs resolved from the environment context profile."""

    context_plugin_ids: tuple[str, ...]
    assembly_options: TaskContextAssemblyOptions
    budget_policy: ContextBudgetPolicy | None
    engine_preset: str
    engine_ref: str | None
    strict: bool


@dataclass(frozen=True, slots=True)
class HostOrchestrationPlanningSpec:
    """Planner and classifier inputs resolved from the environment orchestration profile."""

    planner_kind: str | None
    classifier_kind: str | None
    multi_agent_order: MultiAgentOrder
    coordination_pattern: str | None
    intent_routes: tuple[IntentRoute, ...]
    planner_prompt_id: str
    classifier_prompt_id: str
    planner_parse_retries: int
    llm_adapter: LLMAdapter | None
    planner_llm_adapter: LLMAdapter | None
    graph_topology_source: HostGraphTopologySource | None
    delegation_budget: DelegationBudgetPolicy | None


@dataclass(frozen=True, slots=True)
class HostOrchestrationEnvironment:
    """Execution-engine inputs resolved from the environment profile by Tier-3."""

    execution_mode: ExecutionMode
    retry_policy_name: str | None
    long_running_scheduler_enabled: bool
    attempt_lifecycle_persistence_provider: AttemptLifecyclePersistenceProvider | None
    execution_lineage_persistence_provider: ExecutionLineagePersistenceProvider | None
    execution_terminal_persistence_provider: ExecutionTerminalPersistenceProvider | None
    max_parallel_nodes: int | None
    max_inflight_nodes: int | None
    max_delegation_depth: int | None
    max_run_retries: int
    merge_strategy: MergeStrategy
    allow_dynamic_replan: bool
    emit_coordination_advisory: bool
    denied_planner_model_ids: tuple[str, ...]
    planner_model_id: str | None


__all__ = [
    "HostContextCompositionSpec",
    "HostOrchestrationEnvironment",
    "HostOrchestrationPlanningSpec",
]
