# © Artur Czarnecki. All rights reserved.

"""ERL reliability case lifecycle contract tests."""

from __future__ import annotations

import pytest

from intergrax.contracts.enterprise_reliability.case_lifecycle import (
    ReliabilityCaseLifecycleContextError,
    ReliabilityCaseLifecycleRefs,
    ReliabilityCaseLifecycleState,
    ReliabilityCaseLifecycleTransitionError,
    assert_reliability_case_lifecycle_transition,
    initial_reliability_case_lifecycle,
)

pytestmark = pytest.mark.unit


def _refs_for_state(state: ReliabilityCaseLifecycleState) -> ReliabilityCaseLifecycleRefs:
    base = ReliabilityCaseLifecycleRefs(
        contract_id="pay-1",
        uncertainty_state_ref="erl:uncertainty:corr-1",
        evidence_ref="erl:evidence:corr-1",
        resolution_context_ref="erl:resolution:corr-1:pay-1",
        compensation_context_ref="erl:compensation:corr-1:pay-1",
        recovery_context_ref="erl:recovery:corr-1:pay-1",
        governance_result_ref="erl:governance:allow:corr-1:pay-1",
        handoff_request_ref="erl:handoff:corr-1:pay-1",
    )
    if state is ReliabilityCaseLifecycleState.UNKNOWN_DETECTED:
        return ReliabilityCaseLifecycleRefs(
            contract_id=base.contract_id,
            uncertainty_state_ref=base.uncertainty_state_ref,
        )
    if state is ReliabilityCaseLifecycleState.RECONCILIATION_RUNNING:
        return ReliabilityCaseLifecycleRefs(
            contract_id=base.contract_id,
            uncertainty_state_ref=base.uncertainty_state_ref,
        )
    if state is ReliabilityCaseLifecycleState.EVIDENCE_AVAILABLE:
        return ReliabilityCaseLifecycleRefs(
            contract_id=base.contract_id,
            uncertainty_state_ref=base.uncertainty_state_ref,
            evidence_ref=base.evidence_ref,
        )
    if state is ReliabilityCaseLifecycleState.RESOLUTION_PENDING:
        return ReliabilityCaseLifecycleRefs(
            contract_id=base.contract_id,
            uncertainty_state_ref=base.uncertainty_state_ref,
            evidence_ref=base.evidence_ref,
            resolution_context_ref=base.resolution_context_ref,
        )
    if state is ReliabilityCaseLifecycleState.COMPENSATION_PENDING:
        return base.model_copy(
            update={
                "compensation_context_ref": base.compensation_context_ref,
            },
        )
    if state is ReliabilityCaseLifecycleState.RECOVERY_PENDING:
        return ReliabilityCaseLifecycleRefs(
            contract_id=base.contract_id,
            uncertainty_state_ref=base.uncertainty_state_ref,
            evidence_ref=base.evidence_ref,
            resolution_context_ref=base.resolution_context_ref,
            recovery_context_ref=base.recovery_context_ref,
        )
    if state is ReliabilityCaseLifecycleState.GOVERNANCE_PENDING:
        return ReliabilityCaseLifecycleRefs(
            contract_id=base.contract_id,
            uncertainty_state_ref=base.uncertainty_state_ref,
            evidence_ref=base.evidence_ref,
            resolution_context_ref=base.resolution_context_ref,
            recovery_context_ref=base.recovery_context_ref,
        )
    if state is ReliabilityCaseLifecycleState.HANDOFF_READY:
        return ReliabilityCaseLifecycleRefs(
            contract_id=base.contract_id,
            uncertainty_state_ref=base.uncertainty_state_ref,
            evidence_ref=base.evidence_ref,
            resolution_context_ref=base.resolution_context_ref,
            recovery_context_ref=base.recovery_context_ref,
            governance_result_ref=base.governance_result_ref,
        )
    return base


def test_valid_lifecycle_transition_chain() -> None:
    record = initial_reliability_case_lifecycle(
        case_id="case-1",
        correlation_id="corr-1",
        contract_id="pay-1",
        uncertainty_state_ref="erl:uncertainty:corr-1",
    )
    path = [
        ReliabilityCaseLifecycleState.RECONCILIATION_RUNNING,
        ReliabilityCaseLifecycleState.EVIDENCE_AVAILABLE,
        ReliabilityCaseLifecycleState.RESOLUTION_PENDING,
        ReliabilityCaseLifecycleState.RECOVERY_PENDING,
        ReliabilityCaseLifecycleState.GOVERNANCE_PENDING,
        ReliabilityCaseLifecycleState.HANDOFF_READY,
        ReliabilityCaseLifecycleState.CLOSED,
    ]
    current = record.lifecycle_state
    for target in path:
        assert_reliability_case_lifecycle_transition(current, target)
        current = target


def test_invalid_transition_unknown_to_closed_rejected() -> None:
    with pytest.raises(ReliabilityCaseLifecycleTransitionError):
        assert_reliability_case_lifecycle_transition(
            ReliabilityCaseLifecycleState.UNKNOWN_DETECTED,
            ReliabilityCaseLifecycleState.CLOSED,
        )


def test_lifecycle_refs_remain_immutable() -> None:
    from pydantic import ValidationError

    record = initial_reliability_case_lifecycle(
        case_id="case-2",
        correlation_id="corr-2",
        contract_id="pay-1",
        uncertainty_state_ref="erl:uncertainty:corr-2",
    )
    refs_before = record.refs
    with pytest.raises(ValidationError):
        record.refs = ReliabilityCaseLifecycleRefs(
            contract_id="other",
            uncertainty_state_ref="erl:uncertainty:other",
        )
    assert record.refs is refs_before
    assert record.refs.contract_id == "pay-1"


def test_missing_context_for_target_state_fails_closed() -> None:
    from intergrax.contracts.enterprise_reliability.case_lifecycle import (
        assert_reliability_case_lifecycle_refs_for_state,
    )

    refs = ReliabilityCaseLifecycleRefs(
        contract_id="pay-1",
        uncertainty_state_ref="erl:uncertainty:corr-3",
    )
    with pytest.raises(ReliabilityCaseLifecycleContextError):
        assert_reliability_case_lifecycle_refs_for_state(
            ReliabilityCaseLifecycleState.EVIDENCE_AVAILABLE,
            refs,
        )
