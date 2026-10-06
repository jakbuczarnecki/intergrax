# © Artur Czarnecki. All rights reserved.

"""MEANINGFUL_SIDE_EFFECT governance evidence ref projection (persistence truthfulness)."""

from __future__ import annotations

import pytest

from intergrax.contracts.collaborative_work import CollaborativeWorkEnforcementRequest
from intergrax.contracts.governed_execution_governance_evidence import (
    GovernanceEvidencePersistenceOutcome,
)
from intergrax.contracts.meaningful_side_effect import (
    MeaningfulSideEffectKind,
    MeaningfulSideEffectRequest,
)
from intergrax.contracts.runtime_policy import PolicyAction, PolicyDecision
from intergrax.runtime.governance.mse_governance_evidence_projection import (
    mse_governance_evidence_ref,
    mse_governance_evidence_ref_from_persisted_outcome,
)
from tests.unit.runtime.governance.gr3_test_support import default_gr3_identity_bundle

pytestmark = [pytest.mark.unit]


def _sample_request_and_decision() -> tuple[CollaborativeWorkEnforcementRequest, PolicyDecision]:
    task_id, run_id, attempt_id, execution_id = default_gr3_identity_bundle()
    decision = PolicyDecision(
        action=PolicyAction.ALLOW,
        reason="ok",
        policy_rule_id="test.rule",
    )
    side_effect = MeaningfulSideEffectRequest(
        action="external_work.create",
        kinds=(MeaningfulSideEffectKind.MUTATION,),
        side_effect_scope_id="scope",
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
        tenant_id="tenant-a",
        principal_id="principal-a",
    )
    request = CollaborativeWorkEnforcementRequest(
        tenant_id="tenant-a",
        workspace_id="ws",
        operation_id="external_work.create",
        acting_principal_id="principal-a",
        resource_scope="resource",
        meaningful_side_effect_request=side_effect,
    )
    return request, decision


def test_persisted_false_with_evidence_id_produces_no_ref() -> None:
    request, decision = _sample_request_and_decision()
    expected = mse_governance_evidence_ref(request, decision)
    assert expected is not None
    outcome = GovernanceEvidencePersistenceOutcome(
        persisted=False,
        evidence_id=expected.evidence_id,
        error_code="GovernanceEvidencePersistenceError",
    )
    assert (
        mse_governance_evidence_ref_from_persisted_outcome(request, decision, outcome)
        is None
    )


def test_contradictory_success_evidence_id_fails_closed() -> None:
    request, decision = _sample_request_and_decision()
    expected = mse_governance_evidence_ref(request, decision)
    assert expected is not None
    outcome = GovernanceEvidencePersistenceOutcome(
        persisted=True,
        evidence_id=f"{expected.evidence_id}-mismatch",
    )
    assert (
        mse_governance_evidence_ref_from_persisted_outcome(request, decision, outcome)
        is None
    )


def test_idempotency_conflict_outcome_produces_no_ref() -> None:
    request, decision = _sample_request_and_decision()
    expected = mse_governance_evidence_ref(request, decision)
    assert expected is not None
    outcome = GovernanceEvidencePersistenceOutcome(
        persisted=False,
        evidence_id=expected.evidence_id,
        error_code="GovernanceEvidenceIdempotencyConflict",
    )
    assert (
        mse_governance_evidence_ref_from_persisted_outcome(request, decision, outcome)
        is None
    )


def test_persisted_true_matching_evidence_id_produces_ref() -> None:
    request, decision = _sample_request_and_decision()
    expected = mse_governance_evidence_ref(request, decision)
    assert expected is not None
    outcome = GovernanceEvidencePersistenceOutcome(
        persisted=True,
        evidence_id=expected.evidence_id,
    )
    ref = mse_governance_evidence_ref_from_persisted_outcome(request, decision, outcome)
    assert ref is not None
    assert ref.evidence_id == expected.evidence_id
