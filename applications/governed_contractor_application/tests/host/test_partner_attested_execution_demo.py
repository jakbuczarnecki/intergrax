# © Artur Czarnecki. All rights reserved.

"""Partner demo: governed side effect → ExecutionBoundaryEvent → attested ProofReceipt.

Offline, deterministic fake provider — no network. Host owns attestation.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from external_contractor_adapter.external_work_adapter import (
    META_CORRELATION_ID,
    META_IDEMPOTENCY_KEY,
    META_PROVIDER_ID,
    META_SCOPE_DESCRIPTION,
    META_SCOPE_DIGEST,
    ExternalWorkAdapter,
)
from external_contractor_adapter.side_effect_actions import (
    ACTION_ACCEPT_QUOTE,
    ACTION_CREATE_EXTERNAL_WORK,
)
from external_contractor_adapter.tests.fakes.deterministic_external_work import (
    DeterministicExternalWorkFake,
)
from external_contractor_adapter.tests.fakes.deterministic_side_effect_policy import (
    DeterministicMeaningfulSideEffectPolicy,
)
from governed_contractor_application.host.execution_evidence import (
    produce_attested_receipt_for_adapter_result,
)
from intergrax.contracts.actor_identity import ActorIdentity, ActorKind
from intergrax.contracts.external_work import QuoteAcceptanceEvidence
from intergrax.contracts.governed_continuation import ContinuationReason
from intergrax.contracts.governed_proof import EVIDENCE_KIND_QUOTE_ACCEPTANCE
from intergrax.contracts.money import MoneyAmount
from intergrax.contracts.runtime_policy import PolicyAction
from intergrax.contracts.runtime_policy_bundle import (
    PolicyBundleRule,
    build_immutable_runtime_policy_bundle,
)
from intergrax.runtime.attestation.canonical_json import stable_payload_hash
from intergrax.runtime.execution_evidence.attestor import build_deterministic_test_attestor
from intergrax.runtime.execution_evidence.verify import (
    StaticKeyResolver,
    verify_proof_receipt,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_DIGEST = "sha256:" + ("ab" * 32)
_T0 = datetime(2026, 7, 20, 18, 0, 0, tzinfo=timezone.utc)
_TASK_ID = "task-partner-attest"
_RUN_ID = "run-partner-attest"
_PROVIDER_ID = "gec3_deterministic_fake"
_CORR_ID = "corr-partner-attest"
_CREATE_IDEMP = "idem-partner-attest-create"
_ACCEPT_IDEMP = "idem-partner-attest-accept"


def _bundle():
    return build_immutable_runtime_policy_bundle(
        bundle_id="gec-partner-policy",
        version="1.0.0",
        rules=(
            PolicyBundleRule(
                rule_id="fake.meaningful_side_effect.CREATE_EXTERNAL_WORK",
                description="allow create",
                effect="allow",
            ),
            PolicyBundleRule(
                rule_id="fake.meaningful_side_effect.ACCEPT_QUOTE",
                description="allow accept",
                effect="allow",
            ),
        ),
        issued_at=_T0,
    )


def _meta() -> dict[str, object]:
    return {
        META_PROVIDER_ID: _PROVIDER_ID,
        META_SCOPE_DESCRIPTION: "partner attestation demo scope",
        META_SCOPE_DIGEST: _DIGEST,
        META_IDEMPOTENCY_KEY: _CREATE_IDEMP,
        META_CORRELATION_ID: _CORR_ID,
        "external_work.budget_limit": MoneyAmount(
            amount=Decimal("25.00"), currency="USD"
        ),
        "external_work.principal_id": "partner-demo-user",
        "external_work.tenant_id": "partner-demo-tenant",
    }


def _assert_receipt_fields(receipt, *, action: str, evidence_kind: str | None) -> None:
    event = receipt.execution_boundary_event
    att = receipt.host_attestation
    assert event.task_id == _TASK_ID
    assert event.run_id == _RUN_ID
    assert event.provider_id == _PROVIDER_ID
    assert event.action == action
    assert event.correlation_id == _CORR_ID
    assert event.policy.bundle_id == "gec-partner-policy"
    assert event.policy.bundle_version == "1.0.0"
    assert event.policy.bundle_digest.startswith("sha256:")
    assert event.policy.rule_id
    assert event.policy.action is PolicyAction.ALLOW
    assert event.governed_proof.proof_digest.startswith("sha256:")
    assert event.provider_invocation.invocation_id
    assert att.key_id
    assert att.signature
    assert att.payload_digest == stable_payload_hash(event.canonical_payload())
    if evidence_kind is None:
        assert event.governance_evidence is None
    else:
        assert event.governance_evidence is not None
        assert event.governance_evidence.kind == evidence_kind


def test_partner_attested_create_and_accept_lifecycle() -> None:
    bundle = _bundle()
    policy = DeterministicMeaningfulSideEffectPolicy(
        default=PolicyAction.ALLOW,
        policy_bundle=bundle,
    )
    fake = DeterministicExternalWorkFake()
    adapter = ExternalWorkAdapter(fake, side_effect_policy=policy)
    attestor = build_deterministic_test_attestor(
        key_id="governed-contractor-test-host-1",
        clock=lambda: _T0,
        attestation_id="att-partner-demo",
    )
    resolver = StaticKeyResolver({attestor.key_id: attestor.public_key_bytes})

    created = adapter.create_and_map(
        adapter.build_create_request(
            task_id=_TASK_ID,
            run_id=_RUN_ID,
            metadata=_meta(),
        ),
        principal_id="partner-demo-user",
        tenant_id="partner-demo-tenant",
    )
    assert created.used is True
    assert created.proof is not None
    assert created.policy_decision is not None
    assert created.policy_decision.has_attested_policy_bundle_refs()

    create_outcome = produce_attested_receipt_for_adapter_result(
        created,
        attestor=attestor,
        attestation_required=True,
        event_id="ebe-create-fixed",
        receipt_id="rcpt-create-fixed",
        occurred_at=_T0,
    )
    assert create_outcome.execution_succeeded
    assert create_outcome.attestation_succeeded
    assert create_outcome.receipt is not None
    _assert_receipt_fields(
        create_outcome.receipt,
        action=ACTION_CREATE_EXTERNAL_WORK,
        evidence_kind=None,
    )
    create_vr = verify_proof_receipt(create_outcome.receipt, key_resolver=resolver)
    assert create_vr.valid is True

    surfaced = adapter.with_continuation_surface(created, run_id=_RUN_ID)
    assert surfaced.continuation is not None
    assert surfaced.continuation.reason is ContinuationReason.QUOTE

    acceptance = QuoteAcceptanceEvidence.model_validate(
        {
            "acceptance_id": "acc-partner-attest",
            "quote_id": created.quote.quote_id,  # type: ignore[union-attr]
            "quote_version": 1,
            "scope_digest": _DIGEST,
            "actor": ActorIdentity(
                kind=ActorKind.USER,
                actor_id="partner-demo-user",
                tenant_id="partner-demo-tenant",
            ),
            "accepted_at": _T0 + timedelta(minutes=5),
            "hitl_decision_id": "hdec-partner-attest",
            "interrupt_id": "intr-partner-attest",
            "policy_decision_ref": "pol-partner-attest",
        }
    )
    accept_calls_before = fake.accept_calls
    create_calls_before_accept = fake.create_calls
    accepted = adapter.forward_quote_acceptance(
        created.snapshot.correlation,  # type: ignore[union-attr]
        acceptance,
        idempotency_key=_ACCEPT_IDEMP,
        principal_id="partner-demo-user",
        tenant_id="partner-demo-tenant",
    )
    assert accepted.used is True
    assert accepted.proof is not None
    assert accepted.proof.governance_evidence is not None
    assert accepted.proof.governance_evidence.kind == EVIDENCE_KIND_QUOTE_ACCEPTANCE
    assert accepted.proof.idempotency_key == _ACCEPT_IDEMP
    assert fake.accept_calls == accept_calls_before + 1

    accept_outcome = produce_attested_receipt_for_adapter_result(
        accepted,
        attestor=attestor,
        attestation_required=True,
        event_id="ebe-accept-fixed",
        receipt_id="rcpt-accept-fixed",
        occurred_at=_T0 + timedelta(minutes=6),
    )
    assert accept_outcome.attestation_succeeded
    assert accept_outcome.receipt is not None
    _assert_receipt_fields(
        accept_outcome.receipt,
        action=ACTION_ACCEPT_QUOTE,
        evidence_kind=EVIDENCE_KIND_QUOTE_ACCEPTANCE,
    )
    assert accept_outcome.receipt.execution_boundary_event.idempotency_key == _ACCEPT_IDEMP
    accept_vr = verify_proof_receipt(accept_outcome.receipt, key_resolver=resolver)
    assert accept_vr.valid is True
    assert accept_vr.key_id == attestor.key_id

    # Attestation after accept must not re-invoke provider.
    assert fake.accept_calls == accept_calls_before + 1
    assert fake.create_calls == create_calls_before_accept


def test_deny_produces_no_receipt_and_no_provider_create() -> None:
    bundle = _bundle()
    policy = DeterministicMeaningfulSideEffectPolicy(
        default=PolicyAction.DENY,
        policy_bundle=bundle,
    )
    fake = DeterministicExternalWorkFake()
    adapter = ExternalWorkAdapter(fake, side_effect_policy=policy)
    attestor = build_deterministic_test_attestor(clock=lambda: _T0)
    denied = adapter.create_and_map(
        adapter.build_create_request(
            task_id=_TASK_ID,
            run_id=_RUN_ID,
            metadata=_meta(),
        ),
        principal_id="partner-demo-user",
        tenant_id="partner-demo-tenant",
    )
    assert denied.used is False or denied.proof is None
    assert fake.create_calls == 0
    outcome = produce_attested_receipt_for_adapter_result(
        denied,
        attestor=attestor,
        attestation_required=True,
    )
    assert not outcome.attestation_succeeded
    assert outcome.receipt is None
    assert fake.create_calls == 0


def test_attestation_failure_does_not_repeat_provider() -> None:
    bundle = _bundle()
    policy = DeterministicMeaningfulSideEffectPolicy(
        default=PolicyAction.ALLOW,
        policy_bundle=bundle,
    )
    fake = DeterministicExternalWorkFake()
    adapter = ExternalWorkAdapter(fake, side_effect_policy=policy)
    created = adapter.create_and_map(
        adapter.build_create_request(
            task_id=_TASK_ID,
            run_id=_RUN_ID,
            metadata=_meta(),
        ),
        principal_id="partner-demo-user",
        tenant_id="partner-demo-tenant",
    )
    create_calls = fake.create_calls
    accept_calls = fake.accept_calls
    outcome = produce_attested_receipt_for_adapter_result(
        created,
        attestor=None,
        attestation_required=True,
    )
    assert outcome.execution_succeeded
    assert not outcome.attestation_succeeded
    assert outcome.receipt is None
    assert fake.create_calls == create_calls
    assert fake.accept_calls == accept_calls
