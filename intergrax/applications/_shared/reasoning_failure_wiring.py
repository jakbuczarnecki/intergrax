# © Artur Czarnecki. All rights reserved.

"""Reasoning failure taxonomy coverage across planner kinds (AUDIT-IDEAL-7.3)."""

from __future__ import annotations

from intergrax.applications._shared.orchestration_wiring import (
    NexusClassifierKind,
    NexusPlannerKind,
)
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.contracts.reasoning_failure import ReasoningFailureKind

_PLANNER_FAILURE_KINDS: dict[str, tuple[ReasoningFailureKind, ...]] = {
    NexusPlannerKind.DEFAULT.value: (
        ReasoningFailureKind.PLANNER_PARSE_FAILED,
        ReasoningFailureKind.PLANNER_FALLBACK,
        ReasoningFailureKind.PLANNER_VALIDATION_FAILED,
        ReasoningFailureKind.PLANNER_POLICY_BLOCKED,
    ),
    NexusPlannerKind.ENGINE.value: (
        ReasoningFailureKind.PLANNER_PARSE_FAILED,
        ReasoningFailureKind.PLANNER_FALLBACK,
        ReasoningFailureKind.PLANNER_VALIDATION_FAILED,
        ReasoningFailureKind.PLANNER_POLICY_BLOCKED,
    ),
}

_CLASSIFIER_FAILURE_KINDS: dict[str, tuple[ReasoningFailureKind, ...]] = {
    NexusClassifierKind.DEFAULT.value: (ReasoningFailureKind.CLASSIFIER_FALLBACK,),
    NexusClassifierKind.RULES.value: (ReasoningFailureKind.CLASSIFIER_UNSUPPORTED,),
    NexusClassifierKind.LLM.value: (
        ReasoningFailureKind.CLASSIFIER_FALLBACK,
        ReasoningFailureKind.CLASSIFIER_UNSUPPORTED,
    ),
}


def resolve_reasoning_failure_taxonomy(env: ApplicationEnvironmentProfile) -> dict[str, list[str]]:
    """Return expected failure kinds keyed by orchestration collaborator kind."""
    orch = env.orchestration_profile
    planner_kind = orch.planner_kind or NexusPlannerKind.DEFAULT.value
    classifier_kind = orch.classifier_kind or NexusClassifierKind.DEFAULT.value
    planner_kinds = _PLANNER_FAILURE_KINDS.get(
        planner_kind,
        _PLANNER_FAILURE_KINDS[NexusPlannerKind.DEFAULT.value],
    )
    classifier_kinds = _CLASSIFIER_FAILURE_KINDS.get(
        classifier_kind,
        _CLASSIFIER_FAILURE_KINDS[NexusClassifierKind.DEFAULT.value],
    )
    return {
        "planner": [item.value for item in planner_kinds],
        "classifier": [item.value for item in classifier_kinds],
    }


def reasoning_failure_taxonomy_complete(env: ApplicationEnvironmentProfile) -> bool:
    taxonomy = resolve_reasoning_failure_taxonomy(env)
    return bool(taxonomy.get("planner")) and bool(taxonomy.get("classifier"))
