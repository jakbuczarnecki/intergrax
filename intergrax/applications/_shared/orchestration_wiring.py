# © Artur Czarnecki. All rights reserved.

"""Neutral orchestration profile resolution for Tier-3 hosts (planner/classifier in EE)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.contracts.orchestration_enums import MergeStrategy, MultiAgentOrder
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter


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
    """Resolved orchestration knobs for Execution Engine composition."""

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


def orchestration_requires_llm_adapter(env: ApplicationEnvironmentProfile) -> bool:
    """Return whether orchestration wiring must materialize an LLM adapter."""
    planner_kind = _normalize_planner_kind(env.orchestration_profile.planner_kind)
    classifier_kind = _normalize_classifier_kind(env.orchestration_profile.classifier_kind)
    return planner_kind is NexusPlannerKind.ENGINE or classifier_kind is NexusClassifierKind.LLM


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
    return env.orchestration_profile.max_parallel_nodes


def resolve_max_inflight_nodes(env: ApplicationEnvironmentProfile) -> int | None:
    return env.orchestration_profile.max_inflight_nodes
