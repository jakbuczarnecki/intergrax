# © Artur Czarnecki. All rights reserved.

"""UCA-3 — acquisition request/result contract invariants."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from intergrax.contracts.capability_acquisition.acquisition_evidence import (
    CapabilityAcquisitionEvidence,
)
from intergrax.contracts.capability_acquisition.acquisition_outcome import (
    CapabilityAcquisitionOutcome,
)
from intergrax.contracts.capability_acquisition.acquisition_reason_code import (
    CapabilityAcquisitionReasonCode,
)
from intergrax.contracts.capability_acquisition.acquisition_request import (
    CapabilityAcquisitionRequest,
    derive_capability_acquisition_request_id,
)
from intergrax.contracts.capability_acquisition.acquisition_result import (
    CapabilityAcquisitionResult,
)
from intergrax.contracts.capability_catalog.capability_gap import CapabilityGap
from intergrax.contracts.capability_catalog.discovery_completion import (
    build_discovery_completion,
)
from intergrax.contracts.capability_catalog.federation import (
    CapabilityCatalogFederationCompleteness,
)
from intergrax.contracts.capability_catalog.kind import CapabilityKind
from intergrax.contracts.capability_catalog.need import CapabilityNeed
from intergrax.contracts.capability_acquisition.acquisition_authorization import (
    CapabilityAcquisitionAuthorizationOutcome,
    CapabilityAcquisitionAuthorizationResult,
)
from intergrax.contracts.capability_acquisition.strategy_selection import (
    CapabilityAcquisitionStrategySelection,
    CapabilityAcquisitionStrategySelectionOutcome,
)
from intergrax.contracts.policy_action import PolicyAction
from intergrax.contracts.runtime_policy import PolicyDecision
from pydantic import ValidationError

pytestmark = pytest.mark.unit

_CREATED = datetime(2026, 9, 20, 12, 0, tzinfo=UTC)


def _gap() -> CapabilityGap:
    completion = build_discovery_completion(
        need_id="need-1",
        discovery_correlation_id="corr-1",
        federation_completeness=CapabilityCatalogFederationCompleteness.COMPLETE,
        created_at=_CREATED,
    )
    return CapabilityGap.from_discovery_completion(completion)


def test_succeeded_requires_handoff_evidence() -> None:
    with pytest.raises(ValueError, match="acquisition evidence"):
        CapabilityAcquisitionResult(
            request_id="capability-acquisition-request:gap:nonce",
            gap_id="capability-gap:need-1:corr-1",
            strategy_id="s1",
            outcome=CapabilityAcquisitionOutcome.SUCCEEDED,
            reason_code=CapabilityAcquisitionReasonCode.NONE,
            started_at=_CREATED,
            completed_at=_CREATED,
        )


def test_request_rejects_mismatched_need_id() -> None:
    gap = _gap()
    need = CapabilityNeed(need_id="other-need", kinds=(CapabilityKind.TOOL,))
    with pytest.raises(ValueError, match="need_id"):
        CapabilityAcquisitionRequest(
            request_id=derive_capability_acquisition_request_id(
                gap_id=gap.gap_id,
                request_nonce="n1",
            ),
            request_nonce="n1",
            capability_gap=gap,
            capability_need=need,
            requested_at=_CREATED,
        )


def test_success_with_artifact_reference() -> None:
    result = CapabilityAcquisitionResult(
        request_id="capability-acquisition-request:gap:nonce",
        gap_id="capability-gap:need-1:corr-1",
        strategy_id="s1",
        outcome=CapabilityAcquisitionOutcome.SUCCEEDED,
        reason_code=CapabilityAcquisitionReasonCode.NONE,
        started_at=_CREATED,
        completed_at=_CREATED,
        evidence=CapabilityAcquisitionEvidence(artifact_reference="artifact://x"),
    )
    assert result.evidence is not None


def test_authorization_permitted_with_deny_policy_invalid() -> None:
    with pytest.raises(ValidationError):
        CapabilityAcquisitionAuthorizationResult(
            outcome=CapabilityAcquisitionAuthorizationOutcome.PERMITTED,
            policy_decision=PolicyDecision(action=PolicyAction.DENY),
        )


def test_authorization_blocked_with_allow_policy_invalid() -> None:
    with pytest.raises(ValidationError):
        CapabilityAcquisitionAuthorizationResult(
            outcome=CapabilityAcquisitionAuthorizationOutcome.BLOCKED,
            policy_decision=PolicyDecision(action=PolicyAction.ALLOW),
        )


def test_authorization_hitl_requires_human_or_escalate() -> None:
    CapabilityAcquisitionAuthorizationResult(
        outcome=CapabilityAcquisitionAuthorizationOutcome.REQUIRES_HITL,
        policy_decision=PolicyDecision(action=PolicyAction.REQUIRE_HUMAN),
    )
    CapabilityAcquisitionAuthorizationResult(
        outcome=CapabilityAcquisitionAuthorizationOutcome.REQUIRES_HITL,
        policy_decision=PolicyDecision(action=PolicyAction.ESCALATE),
    )
    with pytest.raises(ValidationError):
        CapabilityAcquisitionAuthorizationResult(
            outcome=CapabilityAcquisitionAuthorizationOutcome.REQUIRES_HITL,
            policy_decision=PolicyDecision(action=PolicyAction.DENY),
        )


def test_selection_selected_requires_strategy_id() -> None:
    with pytest.raises(ValidationError):
        CapabilityAcquisitionStrategySelection(
            outcome=CapabilityAcquisitionStrategySelectionOutcome.SELECTED,
            strategy_id=None,
            reason_code=CapabilityAcquisitionReasonCode.NONE,
        )


def test_selection_non_selected_forbids_strategy_id() -> None:
    with pytest.raises(ValidationError):
        CapabilityAcquisitionStrategySelection(
            outcome=CapabilityAcquisitionStrategySelectionOutcome.CONFLICT,
            strategy_id="x",
            reason_code=CapabilityAcquisitionReasonCode.SELECTION_CONFLICT,
        )
    with pytest.raises(ValidationError):
        CapabilityAcquisitionStrategySelection(
            outcome=CapabilityAcquisitionStrategySelectionOutcome.NO_STRATEGY,
            strategy_id="x",
            reason_code=CapabilityAcquisitionReasonCode.NO_STRATEGY,
        )


def test_selection_selected_valid() -> None:
    selection = CapabilityAcquisitionStrategySelection(
        outcome=CapabilityAcquisitionStrategySelectionOutcome.SELECTED,
        strategy_id="strategy-a",
        reason_code=CapabilityAcquisitionReasonCode.NONE,
    )
    assert selection.strategy_id == "strategy-a"
