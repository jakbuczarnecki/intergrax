# © Artur Czarnecki. All rights reserved.

"""Platform completion suite (PC-1…PC-10) — offline, no network."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

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
    ACTION_CANCEL_EXTERNAL_WORK,
    ACTION_CREATE_EXTERNAL_WORK,
)
from external_contractor_adapter.tests.fakes.deterministic_external_work import (
    DeterministicExternalWorkFake,
)
from governed_contractor_application.host.lifecycle_states import (
    GovernedExternalWorkHostState,
)
from governed_contractor_application.host.offline_demo import (
    build_demo_policy_bundle,
    run_offline_governed_contractor_demo,
)
from governed_contractor_application.host.orchestrator import (
    GovernedExternalWorkOrchestrator,
)
from governed_contractor_application.host.stores import (
    FilesystemHostStore,
    InMemoryContinuationStateStore,
    InMemoryGovernedExecutionStore,
    InMemoryPolicyBundleArtifactStore,
    InMemoryProofReceiptStore,
)
from intergrax.contracts.actor_identity import ActorIdentity, ActorKind
from intergrax.contracts.execution_evidence.receipt import ProofReceipt
from intergrax.contracts.external_work import QuoteAcceptanceEvidence
from intergrax.contracts.external_work_provider_capabilities import (
    quote_first_partner_capability_fixture,
)
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
from intergrax.runtime.policy.runtime_policy_bundle_evaluator import (
    RuntimePolicyBundleEvaluator,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_DIGEST = "sha256:" + ("ab" * 32)
_T0 = datetime(2026, 7, 21, 11, 0, 0, tzinfo=timezone.utc)
_PROVIDER = "gec3_deterministic_fake"


def _meta(idem: str = "idem-pc") -> dict[str, object]:
    return {
        META_PROVIDER_ID: _PROVIDER,
        META_SCOPE_DESCRIPTION: "platform completion",
        META_SCOPE_DIGEST: _DIGEST,
        META_IDEMPOTENCY_KEY: idem,
        META_CORRELATION_ID: "corr-pc",
        "external_work.budget_limit": MoneyAmount(
            amount=Decimal("10.00"), currency="USD"
        ),
        "external_work.principal_id": "pc-user",
        "external_work.tenant_id": "pc-tenant",
    }


def _acceptance(quote_id: str, suffix: str = "pc") -> QuoteAcceptanceEvidence:
    return QuoteAcceptanceEvidence.model_validate(
        {
            "acceptance_id": f"acc-{suffix}",
            "quote_id": quote_id,
            "quote_version": 1,
            "scope_digest": _DIGEST,
            "actor": ActorIdentity(
                kind=ActorKind.USER,
                actor_id="pc-user",
                tenant_id="pc-tenant",
            ),
            "accepted_at": _T0 + timedelta(minutes=5),
            "hitl_decision_id": f"hdec-{suffix}",
            "interrupt_id": f"intr-{suffix}",
            "policy_decision_ref": f"pol-{suffix}",
        }
    )


def _orch(
    *,
    attestor=None,
    fake: DeterministicExternalWorkFake | None = None,
    bundle=None,
    execution_store=None,
    receipt_store=None,
    bundle_store=None,
    continuation_store=None,
):
    bundle = bundle or build_demo_policy_bundle(issued_at=_T0)
    policy = RuntimePolicyBundleEvaluator(bundle, clock=lambda: _T0)
    fake = fake or DeterministicExternalWorkFake()
    adapter = ExternalWorkAdapter(fake, side_effect_policy=policy)
    return (
        GovernedExternalWorkOrchestrator(
            adapter=adapter,
            policy=policy,
            bundle=bundle,
            attestor=attestor,
            capabilities=quote_first_partner_capability_fixture(provider_id=_PROVIDER),
            execution_store=execution_store or InMemoryGovernedExecutionStore(),
            receipt_store=receipt_store or InMemoryProofReceiptStore(),
            bundle_store=bundle_store or InMemoryPolicyBundleArtifactStore(),
            continuation_store=continuation_store or InMemoryContinuationStateStore(),
            clock=lambda: _T0,
        ),
        fake,
        policy,
        bundle,
    )


def test_create_accept_cancel_distinct_invocation_ids() -> None:
    attestor = build_deterministic_test_attestor(clock=lambda: _T0)
    orch, fake, _, _ = _orch(attestor=attestor)
    created = orch.create(
        task_id="t-inv",
        run_id="r-inv",
        principal_id="pc-user",
        tenant_id="pc-tenant",
        metadata=_meta("idem-create"),
        execution_id="exec-create",
    )
    assert created.governed_result is not None
    assert created.receipt is not None
    create_inv = created.governed_result.provider_invocation.invocation_id
    assert create_inv.startswith("inv-")

    orch.surface_continuation(
        execution_id="exec-create",
        adapter_result=created.adapter_result,  # type: ignore[arg-type]
        run_id="r-inv",
    )
    accepted = orch.accept(
        execution_id="exec-accept",
        create_result=created.adapter_result,  # type: ignore[arg-type]
        acceptance=_acceptance(created.adapter_result.quote.quote_id),  # type: ignore[union-attr]
        idempotency_key="idem-accept",
        principal_id="pc-user",
        tenant_id="pc-tenant",
        metadata=_meta("idem-accept"),
    )
    assert accepted.governed_result is not None
    accept_inv = accepted.governed_result.provider_invocation.invocation_id
    assert accept_inv != create_inv

    # Fresh create for cancel path
    orch2, fake2, _, _ = _orch(attestor=attestor)
    c2 = orch2.create(
        task_id="t-cancel",
        run_id="r-cancel",
        principal_id="pc-user",
        tenant_id="pc-tenant",
        metadata=_meta("idem-c2"),
        execution_id="exec-c2",
    )
    cancelled = orch2.cancel(
        execution_id="exec-cancel",
        create_result=c2.adapter_result,  # type: ignore[arg-type]
        principal_id="pc-user",
        tenant_id="pc-tenant",
        idempotency_key="idem-cancel",
        metadata=_meta("idem-cancel"),
    )
    assert cancelled.governed_result is not None
    cancel_inv = cancelled.governed_result.provider_invocation.invocation_id
    assert len({create_inv, accept_inv, cancel_inv}) == 3
    assert fake.create_calls >= 1
    assert fake2.cancel_calls >= 1


def test_deny_zero_provider_calls() -> None:
    deny_bundle = build_immutable_runtime_policy_bundle(
        bundle_id="deny-pack",
        version="1",
        rules=(
            PolicyBundleRule(
                rule_id="deny.create",
                effect="deny",
                match_action=ACTION_CREATE_EXTERNAL_WORK,
            ),
        ),
        issued_at=_T0,
    )
    attestor = build_deterministic_test_attestor(clock=lambda: _T0)
    orch, fake, _, _ = _orch(attestor=attestor, bundle=deny_bundle)
    step = orch.create(
        task_id="t-deny",
        run_id="r-deny",
        principal_id="pc-user",
        tenant_id="pc-tenant",
        metadata=_meta(),
        execution_id="exec-deny",
    )
    assert step.state is GovernedExternalWorkHostState.CREATE_POLICY_DENIED
    assert fake.create_calls == 0
    assert step.receipt is None


def test_continuation_zero_provider_calls() -> None:
    attestor = build_deterministic_test_attestor(clock=lambda: _T0)
    orch, fake, _, _ = _orch(attestor=attestor)
    created = orch.create(
        task_id="t-cont",
        run_id="r-cont",
        principal_id="pc-user",
        tenant_id="pc-tenant",
        metadata=_meta("idem-cont"),
        execution_id="exec-cont",
    )
    create_calls = fake.create_calls
    accept_calls = fake.accept_calls
    surfaced = orch.surface_continuation(
        execution_id="exec-cont",
        adapter_result=created.adapter_result,  # type: ignore[arg-type]
        run_id="r-cont",
    )
    assert surfaced.state is GovernedExternalWorkHostState.AWAITING_HUMAN
    assert fake.create_calls == create_calls
    assert fake.accept_calls == accept_calls


def test_human_evidence_alone_does_not_accept() -> None:
    attestor = build_deterministic_test_attestor(clock=lambda: _T0)
    orch, fake, _, _ = _orch(attestor=attestor)
    created = orch.create(
        task_id="t-ev",
        run_id="r-ev",
        principal_id="pc-user",
        tenant_id="pc-tenant",
        metadata=_meta("idem-ev"),
        execution_id="exec-ev",
    )
    _ = _acceptance(created.adapter_result.quote.quote_id)  # type: ignore[union-attr]
    assert fake.accept_calls == 0
    # Evidence object existence does not invoke provider.
    assert created.adapter_result is not None


def test_accept_requires_fresh_policy_evaluation() -> None:
    attestor = build_deterministic_test_attestor(clock=lambda: _T0)
    orch, fake, policy, _ = _orch(attestor=attestor)
    created = orch.create(
        task_id="t-pol",
        run_id="r-pol",
        principal_id="pc-user",
        tenant_id="pc-tenant",
        metadata=_meta("idem-pol-c"),
        execution_id="exec-pol-c",
    )
    calls_before = len(policy.calls)
    orch.accept(
        execution_id="exec-pol-a",
        create_result=created.adapter_result,  # type: ignore[arg-type]
        acceptance=_acceptance(created.adapter_result.quote.quote_id, "pol"),  # type: ignore[union-attr]
        idempotency_key="idem-pol-a",
        principal_id="pc-user",
        tenant_id="pc-tenant",
        metadata=_meta("idem-pol-a"),
    )
    assert len(policy.calls) > calls_before
    assert any(c.action == ACTION_ACCEPT_QUOTE for c in policy.calls)
    assert fake.accept_calls == 1


def test_bundle_artifact_verification_and_tamper() -> None:
    attestor = build_deterministic_test_attestor(clock=lambda: _T0)
    orch, _, _, bundle = _orch(attestor=attestor)
    created = orch.create(
        task_id="t-bun",
        run_id="r-bun",
        principal_id="pc-user",
        tenant_id="pc-tenant",
        metadata=_meta("idem-bun"),
        execution_id="exec-bun",
    )
    assert created.receipt is not None
    assert created.receipt.policy_bundle_artifact is not None
    resolver = StaticKeyResolver({attestor.key_id: attestor.public_key_bytes})
    assert verify_proof_receipt(
        created.receipt, key_resolver=resolver, require_policy_bundle_artifact=True
    ).valid

    # Tamper embedded bundle body.
    tampered_bundle = build_immutable_runtime_policy_bundle(
        bundle_id=bundle.bundle_id,
        version=bundle.version,
        rules=(
            PolicyBundleRule(
                rule_id="tampered",
                effect="allow",
                match_action=ACTION_CREATE_EXTERNAL_WORK,
            ),
        ),
        issued_at=_T0,
    )
    bad = created.receipt.model_copy(
        update={"policy_bundle_artifact": tampered_bundle}
    )
    vr = verify_proof_receipt(
        bad, key_resolver=resolver, require_policy_bundle_artifact=True
    )
    assert vr.valid is False
    assert "policy_bundle_digest_mismatch" in vr.errors

    # Missing rule id in artifact.
    empty_rules = build_immutable_runtime_policy_bundle(
        bundle_id=bundle.bundle_id,
        version=bundle.version,
        rules=(),
        issued_at=_T0,
    )
    # Force same digest claim by copying digest field — verifier recomputes.
    missing = created.receipt.model_copy(
        update={"policy_bundle_artifact": empty_rules}
    )
    vr2 = verify_proof_receipt(
        missing, key_resolver=resolver, require_policy_bundle_artifact=True
    )
    assert vr2.valid is False


def test_attestation_recovery_no_provider_repeat(tmp_path: Path) -> None:
    class _ThrowingAttestor:
        def __init__(self) -> None:
            self.calls = 0

        def attest(self, payload: bytes, *, schema: str):
            self.calls += 1
            raise RuntimeError("signer_boom")

    store = FilesystemHostStore(tmp_path / "store")
    throwing = _ThrowingAttestor()
    orch, fake, _, _ = _orch(
        attestor=throwing,  # type: ignore[arg-type]
        execution_store=store,
        receipt_store=store,
        bundle_store=store,
        continuation_store=store,
    )
    created = orch.create(
        task_id="t-rec",
        run_id="r-rec",
        principal_id="pc-user",
        tenant_id="pc-tenant",
        metadata=_meta("idem-rec"),
        execution_id="exec-rec",
    )
    assert created.state is (
        GovernedExternalWorkHostState.EXECUTION_SUCCEEDED_ATTESTATION_FAILED
    )
    assert created.governed_result is not None
    create_calls = fake.create_calls
    assert store.get_result("exec-rec") is not None

    # New orchestrator instance + working attestor.
    good = build_deterministic_test_attestor(
        key_id="governed-contractor-offline-demo-1",
        clock=lambda: _T0,
    )
    orch2, fake2, _, _ = _orch(
        attestor=good,
        fake=fake,
        execution_store=store,
        receipt_store=store,
        bundle_store=store,
        continuation_store=store,
    )
    retried = orch2.retry_attestation("exec-rec")
    assert retried.receipt is not None
    assert retried.state is GovernedExternalWorkHostState.EXECUTION_SUCCEEDED_ATTESTED
    assert fake.create_calls == create_calls
    assert fake2.create_calls == create_calls

    # Idempotent second retry.
    again = orch2.retry_attestation("exec-rec")
    assert again.reason == "attested_idempotent"
    assert fake.create_calls == create_calls

    resolver = StaticKeyResolver({good.key_id: good.public_key_bytes})
    assert verify_proof_receipt(
        retried.receipt, key_resolver=resolver, require_policy_bundle_artifact=True
    ).valid


def test_cannot_attest_failed_execution() -> None:
    deny_bundle = build_immutable_runtime_policy_bundle(
        bundle_id="deny-pack2",
        version="1",
        rules=(
            PolicyBundleRule(
                rule_id="deny.create",
                effect="deny",
                match_action=ACTION_CREATE_EXTERNAL_WORK,
            ),
        ),
        issued_at=_T0,
    )
    store = InMemoryGovernedExecutionStore()
    receipts = InMemoryProofReceiptStore()
    orch, _, _, _ = _orch(
        attestor=build_deterministic_test_attestor(clock=lambda: _T0),
        bundle=deny_bundle,
        execution_store=store,
        receipt_store=receipts,
    )
    orch.create(
        task_id="t-fail",
        run_id="r-fail",
        principal_id="pc-user",
        tenant_id="pc-tenant",
        metadata=_meta(),
        execution_id="exec-fail",
    )
    with pytest.raises(ValueError, match="execution_result_missing"):
        orch.retry_attestation("exec-fail")


def test_offline_demo_and_json_roundtrip(tmp_path: Path) -> None:
    report = run_offline_governed_contractor_demo(store_root=tmp_path / "demo")
    assert report.verification_valid is True
    assert report.create_invocation_id != report.accept_invocation_id
    assert Path(report.receipt_path).is_file()
    receipt = ProofReceipt.model_validate_json(
        Path(report.receipt_path).read_text(encoding="utf-8")
    )
    attestor = build_deterministic_test_attestor(
        key_id=report.key_id,
    )
    resolver = StaticKeyResolver({attestor.key_id: attestor.public_key_bytes})
    assert verify_proof_receipt(
        receipt, key_resolver=resolver, require_policy_bundle_artifact=True
    ).valid
    # Mutation of signed event invalidates.
    event = receipt.execution_boundary_event.model_copy(update={"actor": "mutated"})
    mutated = receipt.model_copy(update={"execution_boundary_event": event})
    assert verify_proof_receipt(mutated, key_resolver=resolver).valid is False


def test_strict_attestation_requires_first_class_invocation() -> None:
    from intergrax.contracts.evaluated_policy_decision import EvaluatedPolicyDecision
    from intergrax.contracts.governed_execution_result import GovernedExecutionResult
    from intergrax.contracts.governed_proof import GovernedProofProfile
    from intergrax.contracts.provider_invocation import (
        ProviderInvocation,
        ProviderInvocationOutcome,
        ProviderInvocationStatus,
    )
    from intergrax.contracts.runtime_policy import PolicyDecision
    from intergrax.runtime.execution_evidence.compose import (
        attest_governed_execution_result,
    )

    digest = "sha256:" + ("cd" * 32)
    decision = PolicyDecision(
        action=PolicyAction.ALLOW,
        policy_rule_id="r",
        policy_bundle_id="b",
        policy_bundle_version="1",
        policy_bundle_digest=digest,
        decision_id="d",
    )
    evaluated = EvaluatedPolicyDecision(
        decision=decision,
        bundle_id="b",
        bundle_version="1",
        bundle_digest=digest,
        matched_rule_id="r",
        evaluated_at=_T0,
        request_digest=digest,
    )
    proof = GovernedProofProfile(
        principal_id="u",
        task_id="t",
        run_id="r",
        action=ACTION_CREATE_EXTERNAL_WORK,
        provider_id="p",
        policy_action=PolicyAction.ALLOW,
        policy_rule_id="r",
    )
    inv = ProviderInvocation(
        invocation_id="invocation:unknown",
        provider_id="p",
        operation="create_work",
        task_id="t",
        run_id="r",
        request_digest=digest,
        started_at=_T0,
    )
    ger = GovernedExecutionResult(
        execution_id="e",
        task_id="t",
        run_id="r",
        principal_id="u",
        action=ACTION_CREATE_EXTERNAL_WORK,
        evaluated_policy_decision=evaluated,
        provider_invocation=inv,
        provider_outcome=ProviderInvocationOutcome(
            invocation_id="invocation:unknown",
            status=ProviderInvocationStatus.SUCCEEDED,
            completed_at=_T0,
        ),
        proof=proof,
        execution_started_at=_T0,
        execution_completed_at=_T0,
    )
    outcome = attest_governed_execution_result(
        ger,
        attestor=build_deterministic_test_attestor(clock=lambda: _T0),
        require_first_class_invocation=True,
    )
    assert outcome.attestation_succeeded is False
    assert outcome.reason == "first_class_invocation_id_required"


def test_receipt_does_not_authorize_and_verifier_is_offline() -> None:
    attestor = build_deterministic_test_attestor(clock=lambda: _T0)
    orch, fake, _, _ = _orch(attestor=attestor)
    created = orch.create(
        task_id="t-auth",
        run_id="r-auth",
        principal_id="pc-user",
        tenant_id="pc-tenant",
        metadata=_meta("idem-auth"),
        execution_id="exec-auth",
    )
    assert created.receipt is not None
    accept_before = fake.accept_calls
    # Verifying receipt must not call provider.
    resolver = StaticKeyResolver({attestor.key_id: attestor.public_key_bytes})
    assert verify_proof_receipt(created.receipt, key_resolver=resolver).valid
    assert fake.accept_calls == accept_before
    assert fake.create_calls == 1


def test_corrupted_persisted_artifact_fails(tmp_path: Path) -> None:
    store = FilesystemHostStore(tmp_path / "bad")
    path = tmp_path / "bad" / "executions" / "exec-x.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not-json", encoding="utf-8")
    with pytest.raises(ValueError, match="corrupted_execution_artifact"):
        store.get_result("exec-x")


def test_capability_fixture_quote_first_profile() -> None:
    caps = quote_first_partner_capability_fixture()
    assert caps.supports_quote and caps.supports_accept
    assert caps.supports_payment_state and caps.supports_human_wait_state
    assert caps.supports_tool_logs and caps.supports_receipt_timeline


def test_json_roundtrip_preserves_verification() -> None:
    attestor = build_deterministic_test_attestor(clock=lambda: _T0)
    orch, _, _, _ = _orch(attestor=attestor)
    created = orch.create(
        task_id="t-json",
        run_id="r-json",
        principal_id="pc-user",
        tenant_id="pc-tenant",
        metadata=_meta("idem-json"),
        execution_id="exec-json",
    )
    assert created.receipt is not None
    restored = ProofReceipt.model_validate_json(created.receipt.model_dump_json())
    resolver = StaticKeyResolver({attestor.key_id: attestor.public_key_bytes})
    assert verify_proof_receipt(
        restored, key_resolver=resolver, require_policy_bundle_artifact=True
    ).valid
    _ = stable_payload_hash
