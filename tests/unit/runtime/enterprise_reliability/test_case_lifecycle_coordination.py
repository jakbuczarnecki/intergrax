# © Artur Czarnecki. All rights reserved.

"""ERL reliability case lifecycle coordinator tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from intergrax.contracts.enterprise_reliability.case_lifecycle import (
    ReliabilityCaseLifecycleRefs,
    ReliabilityCaseLifecycleState,
    ReliabilityCaseLifecycleTransitionError,
    ReliabilityCaseLifecycleTransitionRequest,
    initial_reliability_case_lifecycle,
)
from intergrax.runtime.enterprise_reliability import transition_reliability_case_lifecycle

pytestmark = pytest.mark.unit

_RUNTIME_SOURCE = (
    Path(__file__).resolve().parents[4]
    / "intergrax"
    / "runtime"
    / "enterprise_reliability"
    / "case_lifecycle_coordination.py"
)


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
    if state in (
        ReliabilityCaseLifecycleState.UNKNOWN_DETECTED,
        ReliabilityCaseLifecycleState.RECONCILIATION_RUNNING,
    ):
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
        return base
    if state in (
        ReliabilityCaseLifecycleState.RECOVERY_PENDING,
        ReliabilityCaseLifecycleState.GOVERNANCE_PENDING,
    ):
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


def test_coordinator_applies_valid_transition() -> None:
    record = initial_reliability_case_lifecycle(
        case_id="case-1",
        correlation_id="corr-1",
        contract_id="pay-1",
        uncertainty_state_ref="erl:uncertainty:corr-1",
    )
    target = ReliabilityCaseLifecycleState.RECONCILIATION_RUNNING
    result = transition_reliability_case_lifecycle(
        ReliabilityCaseLifecycleTransitionRequest(
            record=record,
            target_state=target,
            refs=_refs_for_state(target),
        ),
    )
    assert result.previous_state is ReliabilityCaseLifecycleState.UNKNOWN_DETECTED
    assert result.record.lifecycle_state is target
    assert record.lifecycle_state is ReliabilityCaseLifecycleState.UNKNOWN_DETECTED


def test_coordinator_rejects_invalid_transition() -> None:
    record = initial_reliability_case_lifecycle(
        case_id="case-2",
        correlation_id="corr-2",
        contract_id="pay-1",
        uncertainty_state_ref="erl:uncertainty:corr-2",
    )
    with pytest.raises(ReliabilityCaseLifecycleTransitionError):
        transition_reliability_case_lifecycle(
            ReliabilityCaseLifecycleTransitionRequest(
                record=record,
                target_state=ReliabilityCaseLifecycleState.CLOSED,
                refs=_refs_for_state(ReliabilityCaseLifecycleState.CLOSED),
            ),
        )


def test_coordinator_does_not_execute_business_actions() -> None:
    source = _RUNTIME_SOURCE.read_text(encoding="utf-8")
    forbidden = (
        "execute_external_effect",
        "plan_external_effect",
        "invoke_plugin",
        "EnterpriseReliabilityPluginGateway",
        "ExecutionLifecyclePort",
        "apply_recovery_lifecycle_intent",
        "handoff_recovery_lifecycle",
    )
    for token in forbidden:
        assert token not in source


def test_coordinator_preserves_immutable_refs_on_transition() -> None:
    record = initial_reliability_case_lifecycle(
        case_id="case-3",
        correlation_id="corr-3",
        contract_id="pay-1",
        uncertainty_state_ref="erl:uncertainty:corr-3",
    )
    refs = _refs_for_state(ReliabilityCaseLifecycleState.RECONCILIATION_RUNNING)
    result = transition_reliability_case_lifecycle(
        ReliabilityCaseLifecycleTransitionRequest(
            record=record,
            target_state=ReliabilityCaseLifecycleState.RECONCILIATION_RUNNING,
            refs=refs,
        ),
    )
    assert result.record.refs is refs
    assert record.refs.uncertainty_state_ref == "erl:uncertainty:corr-3"
