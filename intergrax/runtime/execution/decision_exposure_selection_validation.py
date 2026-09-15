# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Pre/post-strategy host validation for Decision exposure selection (I1-A-R1/R2)."""

from __future__ import annotations

from typing import TypeVar

from intergrax.contracts.decision_exposure_selection import (
    DecisionExposureCandidate,
    DecisionExposurePublicationPolicy,
    DecisionExposureSelectionDecision,
    DecisionExposureSelectionFailure,
    DecisionExposureSelectionFailureCode,
    DecisionExposureSelectionStrategy,
    HostPublicationClass,
)
from intergrax.contracts.execution_identity import AttemptId, validate_attempt_id

T = TypeVar("T")


def validate_decision_exposure_candidate_set(
    candidates: tuple[DecisionExposureCandidate[T], ...],
) -> DecisionExposureSelectionFailure | None:
    """Reject mixed execution attempts before any selection strategy runs."""
    if not candidates:
        return None
    attempt_ids: set[AttemptId] = set()
    for candidate in candidates:
        attempt_ids.add(validate_attempt_id(candidate.execution_lineage.attempt_id))
    if len(attempt_ids) != 1:
        return DecisionExposureSelectionFailure(
            reason_code=DecisionExposureSelectionFailureCode.INVALID_CANDIDATE_SET,
            detail="candidates must belong to exactly one effective attempt",
        )
    return None


def validate_single_attempt_partition_for_selection(
    candidates: tuple[DecisionExposureCandidate[T], ...],
) -> AttemptId | DecisionExposureSelectionFailure:
    """Default-selector partition guard (empty set + single-attempt invariant)."""
    if not candidates:
        return DecisionExposureSelectionFailure(
            reason_code=DecisionExposureSelectionFailureCode.NO_ELIGIBLE_TERMINAL_CANDIDATE,
            detail="no candidates supplied for selection",
        )
    mixed = validate_decision_exposure_candidate_set(candidates)
    if mixed is not None:
        return mixed
    return validate_attempt_id(candidates[0].execution_lineage.attempt_id)


def validate_decision_exposure_selection_decision(
    policy: DecisionExposurePublicationPolicy,
    candidates: tuple[DecisionExposureCandidate[T], ...],
    decision: DecisionExposureSelectionDecision[T],
) -> DecisionExposureSelectionFailure | None:
    """Ensure plugin/default selection stays within candidate membership and policy."""
    matched: DecisionExposureCandidate[T] | None = None
    for candidate in candidates:
        if candidate.exposure is decision.selected:
            matched = candidate
            break
    if matched is None:
        return DecisionExposureSelectionFailure(
            reason_code=DecisionExposureSelectionFailureCode.INVARIANT_VIOLATION,
            detail="selected exposure is not a member of the supplied candidate set",
        )
    if matched.host_publication_class is not HostPublicationClass.HOST_TERMINAL_CANDIDATE:
        return DecisionExposureSelectionFailure(
            reason_code=DecisionExposureSelectionFailureCode.INVARIANT_VIOLATION,
            detail="selected candidate is not host-terminal publishable",
        )
    if matched.evaluation_scope not in policy.eligible_terminal_scopes:
        return DecisionExposureSelectionFailure(
            reason_code=DecisionExposureSelectionFailureCode.INVARIANT_VIOLATION,
            detail="selected candidate evaluation scope is not eligible under publication policy",
        )
    return None


def run_validated_decision_exposure_selection(
    strategy: DecisionExposureSelectionStrategy[T],
    policy: DecisionExposurePublicationPolicy,
    candidates: tuple[DecisionExposureCandidate[T], ...],
) -> DecisionExposureSelectionDecision[T] | DecisionExposureSelectionFailure:
    """Invoke strategy then enforce platform trust invariants on the outcome."""
    pre_validation = validate_decision_exposure_candidate_set(candidates)
    if pre_validation is not None:
        return pre_validation
    outcome = strategy.select(policy, candidates)
    if isinstance(outcome, DecisionExposureSelectionFailure):
        return outcome
    validation = validate_decision_exposure_selection_decision(
        policy,
        candidates,
        outcome,
    )
    if validation is not None:
        return validation
    return outcome
