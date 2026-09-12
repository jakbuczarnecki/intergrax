# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""ERL lifecycle coordinator — validates transitions and updates case lifecycle ownership only."""

from __future__ import annotations

from intergrax.contracts.enterprise_reliability.case_lifecycle import (
    ReliabilityCaseLifecycleRecord,
    ReliabilityCaseLifecycleTransitionRequest,
    ReliabilityCaseLifecycleTransitionResult,
    assert_reliability_case_lifecycle_refs_for_state,
    assert_reliability_case_lifecycle_transition,
)


def transition_reliability_case_lifecycle(
    request: ReliabilityCaseLifecycleTransitionRequest,
) -> ReliabilityCaseLifecycleTransitionResult:
    """
    Apply one explicit lifecycle transition after validation.

    Does not invoke reconciliation, resolution, compensation, governance, execution ports,
    or plugins — existing orchestrators remain authoritative for capability work.
    """
    record = request.record
    target = request.target_state
    assert_reliability_case_lifecycle_transition(record.lifecycle_state, target)
    assert_reliability_case_lifecycle_refs_for_state(target, request.refs)
    previous = record.lifecycle_state
    next_record = ReliabilityCaseLifecycleRecord(
        case_id=record.case_id,
        correlation_id=record.correlation_id,
        lifecycle_state=target,
        refs=request.refs,
    )
    return ReliabilityCaseLifecycleTransitionResult(
        previous_state=previous,
        record=next_record,
    )


__all__ = [
    "transition_reliability_case_lifecycle",
]
