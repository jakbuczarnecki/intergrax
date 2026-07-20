# © Artur Czarnecki. All rights reserved.

"""Host attestation + ProofReceipt + offline verification."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from intergrax.contracts.execution_evidence.boundary_event import (
    ExecutionBoundaryEvent,
    GovernedProofSection,
    PolicyDecisionSection,
    ProviderInvocationSection,
)
from intergrax.contracts.governed_proof import compose_governed_proof_profile
from intergrax.contracts.runtime_policy import PolicyAction, PolicyDecision
from intergrax.runtime.execution_evidence.attestor import (
    build_deterministic_test_attestor,
)
from intergrax.runtime.execution_evidence.compose import (
    attest_after_governed_side_effect,
    compose_execution_boundary_event,
    produce_proof_receipt,
)
from intergrax.runtime.execution_evidence.verify import (
    StaticKeyResolver,
    verify_proof_receipt,
)

pytestmark = [pytest.mark.unit]

_T0 = datetime(2026, 7, 20, 18, 0, 0, tzinfo=timezone.utc)
_BUNDLE_DIGEST = "sha256:" + ("ab" * 32)


def _allow_decision() -> PolicyDecision:
    return PolicyDecision(
        action=PolicyAction.ALLOW,
        reason="test",
        policy_rule_id="rule.create",
        policy_bundle_id="bundle-1",
        policy_bundle_version="1.0.0",
        policy_bundle_digest=_BUNDLE_DIGEST,
        decision_id="pol:create",
    )


def _proof():
    return compose_governed_proof_profile(
        principal_id="user-1",
        tenant_id="tenant-1",
        task_id="task-1",
        run_id="run-1",
        action="CREATE_EXTERNAL_WORK",
        resource="scope",
        provider_id="provider-1",
        policy_action=PolicyAction.ALLOW,
        policy_rule_id="rule.create",
        policy_reason="test",
        correlation_id="corr-1",
        idempotency_key="idem-1",
    )


def test_valid_receipt_verifies() -> None:
    attestor = build_deterministic_test_attestor(clock=lambda: _T0)
    event = compose_execution_boundary_event(
        proof=_proof(),
        policy_decision=_allow_decision(),
        provider_operation="create_work",
        invocation_id="ext-1",
        invocation_completed_at=_T0,
        event_id="ebe-fixed",
        occurred_at=_T0,
    )
    receipt = produce_proof_receipt(
        event=event,
        attestor=attestor,
        receipt_id="rcpt-fixed",
    )
    result = verify_proof_receipt(
        receipt,
        key_resolver=StaticKeyResolver({attestor.key_id: attestor.public_key_bytes}),
    )
    assert result.valid
    assert result.schema_valid
    assert result.digest_valid
    assert result.signature_valid
    assert result.key_id == attestor.key_id


@pytest.mark.parametrize(
    "mutator",
    [
        lambda e: e.model_copy(update={"task_id": "task-mutated"}),
        lambda e: e.model_copy(update={"run_id": "run-mutated"}),
        lambda e: e.model_copy(update={"action": "CANCEL_EXTERNAL_WORK"}),
        lambda e: e.model_copy(
            update={
                "policy": e.policy.model_copy(
                    update={"bundle_digest": "sha256:" + ("ff" * 32)}
                )
            }
        ),
        lambda e: e.model_copy(
            update={
                "provider_invocation": e.provider_invocation.model_copy(
                    update={"invocation_id": "mutated"}
                )
            }
        ),
        lambda e: e.model_copy(
            update={
                "governed_proof": e.governed_proof.model_copy(
                    update={"proof_digest": "sha256:" + ("ee" * 32)}
                )
            }
        ),
    ],
)
def test_modified_event_fields_fail_verification(mutator) -> None:
    attestor = build_deterministic_test_attestor(clock=lambda: _T0)
    event = compose_execution_boundary_event(
        proof=_proof(),
        policy_decision=_allow_decision(),
        provider_operation="create_work",
        invocation_id="ext-1",
        invocation_completed_at=_T0,
        event_id="ebe-fixed",
        occurred_at=_T0,
    )
    receipt = produce_proof_receipt(event=event, attestor=attestor, receipt_id="rcpt")
    mutated_event = mutator(event)
    bad = receipt.model_copy(update={"execution_boundary_event": mutated_event})
    result = verify_proof_receipt(
        bad,
        key_resolver=StaticKeyResolver({attestor.key_id: attestor.public_key_bytes}),
    )
    assert not result.valid
    assert not result.digest_valid


def test_modified_signature_fails() -> None:
    attestor = build_deterministic_test_attestor(clock=lambda: _T0)
    event = compose_execution_boundary_event(
        proof=_proof(),
        policy_decision=_allow_decision(),
        provider_operation="create_work",
        invocation_id="ext-1",
        invocation_completed_at=_T0,
        event_id="ebe-fixed",
        occurred_at=_T0,
    )
    receipt = produce_proof_receipt(event=event, attestor=attestor, receipt_id="rcpt")
    bad_att = receipt.host_attestation.model_copy(
        update={"signature": receipt.host_attestation.signature[:-4] + "AAAA"}
    )
    bad = receipt.model_copy(update={"host_attestation": bad_att})
    result = verify_proof_receipt(
        bad,
        key_resolver=StaticKeyResolver({attestor.key_id: attestor.public_key_bytes}),
    )
    assert not result.valid
    assert "signature_invalid" in result.errors or not result.signature_valid


def test_unknown_key_fails() -> None:
    attestor = build_deterministic_test_attestor(clock=lambda: _T0)
    event = compose_execution_boundary_event(
        proof=_proof(),
        policy_decision=_allow_decision(),
        provider_operation="create_work",
        invocation_id="ext-1",
        invocation_completed_at=_T0,
        event_id="ebe-fixed",
        occurred_at=_T0,
    )
    receipt = produce_proof_receipt(event=event, attestor=attestor, receipt_id="rcpt")
    result = verify_proof_receipt(receipt, key_resolver=StaticKeyResolver({}))
    assert not result.valid
    assert "unknown_key_id" in result.errors


def test_missing_policy_bundle_fails_closed() -> None:
    decision = PolicyDecision(action=PolicyAction.ALLOW, reason="x", policy_rule_id="r")
    with pytest.raises(ValueError, match="policy_bundle_identity_missing"):
        compose_execution_boundary_event(
            proof=_proof(),
            policy_decision=decision,
            provider_operation="create_work",
            invocation_id="ext-1",
            invocation_completed_at=_T0,
        )


def test_cannot_compose_event_without_proof_via_orchestrator() -> None:
    outcome = attest_after_governed_side_effect(
        proof=None,
        policy_decision=_allow_decision(),
        provider_operation="create_work",
        invocation_id="ext-1",
        invocation_completed_at=_T0,
        attestor=build_deterministic_test_attestor(clock=lambda: _T0),
        attestation_required=True,
        execution_succeeded=True,
        provider_invoked=True,
    )
    assert outcome.execution_succeeded
    assert not outcome.attestation_succeeded
    assert outcome.receipt is None
    assert outcome.reason == "proof_missing_after_execution"


def test_attestation_failure_does_not_claim_attested() -> None:
    outcome = attest_after_governed_side_effect(
        proof=_proof(),
        policy_decision=_allow_decision(),
        provider_operation="create_work",
        invocation_id="ext-1",
        invocation_completed_at=_T0,
        attestor=None,
        attestation_required=True,
        execution_succeeded=True,
        provider_invoked=True,
    )
    assert outcome.execution_succeeded
    assert not outcome.attestation_succeeded
    assert outcome.receipt is None
    assert outcome.reason == "host_attestor_missing"


def test_execution_did_not_occur_no_receipt() -> None:
    outcome = attest_after_governed_side_effect(
        proof=_proof(),
        policy_decision=_allow_decision(),
        provider_operation="create_work",
        invocation_id="ext-1",
        invocation_completed_at=_T0,
        attestor=build_deterministic_test_attestor(clock=lambda: _T0),
        attestation_required=True,
        execution_succeeded=False,
        provider_invoked=False,
    )
    assert not outcome.execution_succeeded
    assert outcome.receipt is None
    assert outcome.reason == "execution_did_not_occur"
