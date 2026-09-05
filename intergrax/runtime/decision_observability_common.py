# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Shared Decision observability identity and EmitContext lineage validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeVar

from intergrax.contracts.decision_identity import DecisionIdentity
from intergrax.contracts.decision_record import CandidateDecision, candidate_decision_ref
from intergrax.runtime.events.emit_context import EmitContext

T = TypeVar("T")


def validate_positive_int(value: int, label: str) -> int:
    if type(value) is not int or isinstance(value, bool):
        raise TypeError(f"{label} must be int, got {type(value).__name__}")
    if value < 0:
        raise ValueError(f"{label} must be >= 0")
    return value


def validate_decision_version_value(value: int) -> int:
    if type(value) is not int or isinstance(value, bool):
        raise TypeError(f"decision_version must be int, got {type(value).__name__}")
    if value < 1:
        raise ValueError("decision_version must be >= 1")
    return value


@dataclass(frozen=True, slots=True)
class DecisionObservabilityIdentity:
    """Redaction-safe Decision identity fields for RuntimeEvent payloads."""

    decision_id: str
    decision_version: int
    tenant_id: str
    scope_namespace: str
    task_id: str
    run_id: str
    attempt_id: str
    execution_id: str | None
    proposal_branch_id: str | None = None


def decision_observability_identity_from_decision_identity(
    identity: DecisionIdentity,
    *,
    proposal_branch_id: str | None = None,
) -> DecisionObservabilityIdentity:
    if type(identity) is not DecisionIdentity:
        raise TypeError("identity must be DecisionIdentity")
    execution = identity.execution
    return DecisionObservabilityIdentity(
        decision_id=str(identity.decision_id),
        decision_version=identity.version.value,
        tenant_id=identity.tenant_id,
        scope_namespace=identity.scope.namespace,
        task_id=str(execution.task_id),
        run_id=str(execution.run_id),
        attempt_id=str(execution.attempt_id),
        execution_id=(
            str(execution.execution_id) if execution.execution_id is not None else None
        ),
        proposal_branch_id=proposal_branch_id,
    )


def decision_observability_identity_from_candidate(
    candidate: CandidateDecision[T],
) -> DecisionObservabilityIdentity:
    if type(candidate) is not CandidateDecision:
        raise TypeError("candidate must be CandidateDecision")
    proposal_ref = candidate_decision_ref(candidate)
    return decision_observability_identity_from_decision_identity(
        candidate.identity,
        proposal_branch_id=str(proposal_ref.lineage_ref.branch_id),
    )


def validate_emit_context_lineage_for_identity(
    identity: DecisionIdentity,
    ctx: EmitContext,
) -> None:
    if type(identity) is not DecisionIdentity:
        raise TypeError("identity must be DecisionIdentity")
    execution = identity.execution
    if execution.task_id != ctx.task_id:
        raise ValueError("decision execution task_id must match EmitContext.task_id")
    if execution.run_id != ctx.run_id:
        raise ValueError("decision execution run_id must match EmitContext.run_id")
    if execution.attempt_id != ctx.attempt_id:
        raise ValueError("decision execution attempt_id must match EmitContext.attempt_id")
    if execution.execution_id is not None and execution.execution_id != ctx.execution_id:
        raise ValueError(
            "decision execution execution_id must match EmitContext.execution_id",
        )
    if ctx.tenant_id is not None and identity.tenant_id != ctx.tenant_id:
        raise ValueError("decision tenant_id must match EmitContext.tenant_id")


def validate_emit_context_lineage_for_candidate(
    candidate: CandidateDecision[T],
    ctx: EmitContext,
) -> None:
    if type(candidate) is not CandidateDecision:
        raise TypeError("candidate must be CandidateDecision")
    validate_emit_context_lineage_for_identity(candidate.identity, ctx)
