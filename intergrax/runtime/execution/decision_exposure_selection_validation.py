# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Post-strategy host validation for Decision exposure selection (P0-B-D1-I1-A-R1)."""

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

T = TypeVar("T")


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
