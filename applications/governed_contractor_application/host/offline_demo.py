# © Artur Czarnecki. All rights reserved.

"""Reproducible offline governed-contractor demo (PC-8 / FH).

No network. Deterministic fake provider + local Ed25519 attestor.
Exports public verification keys only — never private keys or seeds.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

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
from intergrax.contracts.actor_identity import ActorIdentity, ActorKind
from intergrax.contracts.external_work import QuoteAcceptanceEvidence
from intergrax.contracts.external_work_provider_capabilities import (
    quote_first_partner_capability_fixture,
)
from intergrax.contracts.money import MoneyAmount
from intergrax.contracts.runtime_policy_bundle import (
    PolicyBundleRule,
    build_immutable_runtime_policy_bundle,
)
from intergrax.runtime.execution_evidence.attestor import build_deterministic_test_attestor
from intergrax.runtime.execution_evidence.key_store import (
    DEMO_OFFLINE_KEY_ID,
    FilesystemHostKeyResolver,
    write_demo_mode_marker,
    write_verification_key_artifact,
)
from intergrax.runtime.execution_evidence.verify import verify_proof_receipt
from intergrax.runtime.policy.runtime_policy_bundle_evaluator import (
    RuntimePolicyBundleEvaluator,
)
from governed_contractor_application.host.lifecycle_states import (
    GovernedExternalWorkHostState,
)
from governed_contractor_application.host.orchestrator import (
    GovernedExternalWorkOrchestrator,
)
from governed_contractor_application.host.stores import FilesystemHostStore

_DIGEST = "sha256:" + ("ab" * 32)
_T0 = datetime(2026, 7, 21, 8, 0, 0, tzinfo=timezone.utc)
_PROVIDER_ID = "gec3_deterministic_fake"


def display_relative_path(path: Path, *, cwd: Path | None = None) -> str:
    """Prefer repo/CWD-relative POSIX path for public CLI output."""
    target = Path(path)
    base = Path(cwd) if cwd is not None else Path.cwd()
    try:
        return target.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return target.as_posix()


@dataclass(frozen=True, slots=True)
class OfflineDemoReport:
    task_id: str
    run_id: str
    create_execution_id: str
    accept_execution_id: str
    create_invocation_id: str
    accept_invocation_id: str
    provider_id: str
    action: str
    policy_bundle_id: str
    policy_bundle_version: str
    policy_bundle_digest: str
    policy_rule: str
    policy_action: str
    proof_digest: str
    event_digest: str
    key_id: str
    receipt_path: str
    receipt_absolute_path: str
    verification_key_path: str
    verification_command: str
    verification_valid: bool
    store_root: str
    provider_execution_succeeded: bool
    attestation_succeeded: bool
    state: str
    recovery_command: str | None
    create_calls: int
    accept_calls: int
    cancel_calls: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "run_id": self.run_id,
            "create_execution_id": self.create_execution_id,
            "accept_execution_id": self.accept_execution_id,
            "create_invocation_id": self.create_invocation_id,
            "accept_invocation_id": self.accept_invocation_id,
            "provider_id": self.provider_id,
            "action": self.action,
            "policy_bundle_id": self.policy_bundle_id,
            "policy_bundle_version": self.policy_bundle_version,
            "policy_bundle_digest": self.policy_bundle_digest,
            "policy_rule": self.policy_rule,
            "policy_action": self.policy_action,
            "proof_digest": self.proof_digest,
            "event_digest": self.event_digest,
            "key_id": self.key_id,
            "receipt_path": self.receipt_path,
            "receipt_absolute_path": self.receipt_absolute_path,
            "verification_key_path": self.verification_key_path,
            "verification_command": self.verification_command,
            "verification_valid": self.verification_valid,
            "store_root": self.store_root,
            "provider_execution_succeeded": self.provider_execution_succeeded,
            "attestation_succeeded": self.attestation_succeeded,
            "state": self.state,
            "recovery_command": self.recovery_command,
            "create_calls": self.create_calls,
            "accept_calls": self.accept_calls,
            "cancel_calls": self.cancel_calls,
        }


class _ThrowingAttestor:
    """Demo-only attestor that simulates signer failure after provider success."""

    def __init__(self) -> None:
        self.calls = 0

    def attest(self, payload: bytes, *, schema: str):
        self.calls += 1
        raise RuntimeError("simulated_signing_failure")


def build_demo_policy_bundle(
    *,
    issued_at: datetime | None = None,
) -> Any:
    return build_immutable_runtime_policy_bundle(
        bundle_id="gec-offline-demo-policy",
        version="1.0.0",
        rules=(
            PolicyBundleRule(
                rule_id="demo.CREATE_EXTERNAL_WORK",
                description="allow create",
                effect="allow",
                match_action=ACTION_CREATE_EXTERNAL_WORK,
            ),
            PolicyBundleRule(
                rule_id="demo.ACCEPT_QUOTE",
                description="allow accept",
                effect="allow",
                match_action=ACTION_ACCEPT_QUOTE,
            ),
            PolicyBundleRule(
                rule_id="demo.CANCEL_EXTERNAL_WORK",
                description="allow cancel",
                effect="allow",
                match_action="CANCEL_EXTERNAL_WORK",
            ),
        ),
        issued_at=issued_at or _T0,
    )


def _write_provider_calls(store_root: Path, fake: DeterministicExternalWorkFake) -> Path:
    payload = {
        "create_calls": fake.create_calls,
        "accept_calls": fake.accept_calls,
        "cancel_calls": fake.cancel_calls,
    }
    path = Path(store_root) / "provider_calls.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _meta() -> dict[str, object]:
    return {
        META_PROVIDER_ID: _PROVIDER_ID,
        META_SCOPE_DESCRIPTION: "offline governed contractor demo",
        META_SCOPE_DIGEST: _DIGEST,
        META_IDEMPOTENCY_KEY: "idem-offline-create",
        META_CORRELATION_ID: "corr-offline-demo",
        "external_work.budget_limit": MoneyAmount(
            amount=Decimal("25.00"), currency="USD"
        ),
        "external_work.principal_id": "offline-demo-user",
        "external_work.tenant_id": "offline-demo-tenant",
    }


def _acceptance(quote_id: str) -> QuoteAcceptanceEvidence:
    return QuoteAcceptanceEvidence.model_validate(
        {
            "acceptance_id": "acc-offline-demo",
            "quote_id": quote_id,
            "quote_version": 1,
            "scope_digest": _DIGEST,
            "actor": ActorIdentity(
                kind=ActorKind.USER,
                actor_id="offline-demo-user",
                tenant_id="offline-demo-tenant",
            ),
            "accepted_at": _T0 + timedelta(minutes=5),
            "hitl_decision_id": "hdec-offline-demo",
            "interrupt_id": "intr-offline-demo",
            "policy_decision_ref": "pol-offline-demo",
        }
    )


def run_offline_governed_contractor_demo(
    *,
    store_root: Path,
    task_id: str = "task-offline-demo",
    run_id: str = "run-offline-demo",
    simulate_signing_failure: bool = False,
) -> OfflineDemoReport:
    """Full CREATE -> quote -> human fixture -> ACCEPT -> attested receipt -> verify.

    When ``simulate_signing_failure`` is True, ACCEPT persists GER + EBE but leaves
    attestation failed so a separate process can ``retry-attestation``.
    """
    store_root = Path(store_root)
    store_root.mkdir(parents=True, exist_ok=True)
    store = FilesystemHostStore(store_root)
    bundle = build_demo_policy_bundle()
    policy = RuntimePolicyBundleEvaluator(bundle, clock=lambda: _T0)
    fake = DeterministicExternalWorkFake()
    adapter = ExternalWorkAdapter(fake, side_effect_policy=policy)
    recovery_attestor = build_deterministic_test_attestor(
        key_id=DEMO_OFFLINE_KEY_ID,
        clock=lambda: _T0,
        attestation_id="att-offline-demo",
    )
    # Public verification material is always exported for the known demo key.
    key_path = write_verification_key_artifact(
        store_root,
        key_id=recovery_attestor.key_id,
        public_key_bytes=recovery_attestor.public_key_bytes,
        created_at=_T0,
    )
    write_demo_mode_marker(store_root, key_id=recovery_attestor.key_id)

    create_attestor = recovery_attestor
    accept_attestor: Any = (
        _ThrowingAttestor() if simulate_signing_failure else recovery_attestor
    )

    orch_create = GovernedExternalWorkOrchestrator(
        adapter=adapter,
        policy=policy,
        bundle=bundle,
        attestor=create_attestor,
        capabilities=quote_first_partner_capability_fixture(
            provider_id=_PROVIDER_ID,
        ),
        execution_store=store,
        receipt_store=store,
        bundle_store=store,
        continuation_store=store,
        clock=lambda: _T0,
    )
    meta = _meta()
    created = orch_create.create(
        task_id=task_id,
        run_id=run_id,
        principal_id="offline-demo-user",
        tenant_id="offline-demo-tenant",
        metadata=meta,
        execution_id="exec-offline-create",
        event_id="ebe-offline-create",
        receipt_id="rcpt-offline-create",
    )
    if created.adapter_result is None or created.governed_result is None:
        raise RuntimeError(f"create_failed:{created.reason}")
    if created.state is not GovernedExternalWorkHostState.EXECUTION_SUCCEEDED_ATTESTED:
        raise RuntimeError(f"create_attestation_failed:{created.reason}")

    orch_create.surface_continuation(
        execution_id=created.execution_id or "exec-offline-create",
        adapter_result=created.adapter_result,
        run_id=run_id,
    )
    acceptance = _acceptance(created.adapter_result.quote.quote_id)  # type: ignore[union-attr]

    orch_accept = GovernedExternalWorkOrchestrator(
        adapter=adapter,
        policy=policy,
        bundle=bundle,
        attestor=accept_attestor,
        capabilities=quote_first_partner_capability_fixture(
            provider_id=_PROVIDER_ID,
        ),
        execution_store=store,
        receipt_store=store,
        bundle_store=store,
        continuation_store=store,
        clock=lambda: _T0,
    )
    accepted = orch_accept.accept(
        execution_id="exec-offline-accept",
        create_result=created.adapter_result,
        acceptance=acceptance,
        idempotency_key="idem-offline-accept",
        principal_id="offline-demo-user",
        tenant_id="offline-demo-tenant",
        metadata=meta,
        event_id="ebe-offline-accept",
        receipt_id="rcpt-offline-accept",
    )
    _write_provider_calls(store_root, fake)

    store_disp = display_relative_path(store_root)
    key_disp = display_relative_path(key_path)
    verify_cmd = (
        f"uv run intergrax receipt verify "
        f"{store_disp}/export/accept_receipt.json --store {store_disp}"
    )
    recovery_cmd = (
        f"uv run intergrax external-work retry-attestation "
        f"exec-offline-accept --store {store_disp}"
    )

    if simulate_signing_failure:
        if accepted.state is not (
            GovernedExternalWorkHostState.EXECUTION_SUCCEEDED_ATTESTATION_FAILED
        ):
            raise RuntimeError(f"expected_signer_failure:{accepted.reason}")
        if accepted.receipt is not None:
            raise RuntimeError("receipt_unexpected_after_signer_failure")
        if accepted.governed_result is None:
            raise RuntimeError("ger_missing_after_signer_failure")
        export_dir = store_root / "export"
        export_dir.mkdir(parents=True, exist_ok=True)
        report = OfflineDemoReport(
            task_id=task_id,
            run_id=run_id,
            create_execution_id=created.execution_id or "",
            accept_execution_id=accepted.execution_id or "exec-offline-accept",
            create_invocation_id=created.governed_result.provider_invocation.invocation_id,
            accept_invocation_id=accepted.governed_result.provider_invocation.invocation_id,
            provider_id=_PROVIDER_ID,
            action=ACTION_ACCEPT_QUOTE,
            policy_bundle_id=bundle.bundle_id,
            policy_bundle_version=bundle.version,
            policy_bundle_digest=bundle.canonical_digest,
            policy_rule=accepted.governed_result.evaluated_policy_decision.matched_rule_id,
            policy_action=accepted.governed_result.evaluated_policy_decision.decision.action.value,
            proof_digest="",
            event_digest="",
            key_id=recovery_attestor.key_id,
            receipt_path="",
            receipt_absolute_path="",
            verification_key_path=key_disp,
            verification_command=verify_cmd,
            verification_valid=False,
            store_root=store_disp,
            provider_execution_succeeded=True,
            attestation_succeeded=False,
            state=accepted.state.value,
            recovery_command=recovery_cmd,
            create_calls=fake.create_calls,
            accept_calls=fake.accept_calls,
            cancel_calls=fake.cancel_calls,
        )
        (store_root / "demo_report.json").write_text(
            json.dumps(report.as_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return report

    if accepted.receipt is None or accepted.governed_result is None:
        raise RuntimeError(f"accept_attestation_failed:{accepted.reason}")
    export_path = store_root / "export" / "accept_receipt.json"
    export_path.parent.mkdir(parents=True, exist_ok=True)
    export_path.write_text(
        accepted.receipt.model_dump_json(indent=2), encoding="utf-8"
    )
    resolver = FilesystemHostKeyResolver(store_root)
    vr = verify_proof_receipt(
        accepted.receipt,
        key_resolver=resolver,
        require_policy_bundle_artifact=True,
    )
    ger = accepted.governed_result
    event = accepted.receipt.execution_boundary_event
    receipt_rel = display_relative_path(export_path)
    report = OfflineDemoReport(
        task_id=task_id,
        run_id=run_id,
        create_execution_id=created.execution_id or "",
        accept_execution_id=accepted.execution_id or "",
        create_invocation_id=created.governed_result.provider_invocation.invocation_id,
        accept_invocation_id=ger.provider_invocation.invocation_id,
        provider_id=_PROVIDER_ID,
        action=ACTION_ACCEPT_QUOTE,
        policy_bundle_id=bundle.bundle_id,
        policy_bundle_version=bundle.version,
        policy_bundle_digest=bundle.canonical_digest,
        policy_rule=ger.evaluated_policy_decision.matched_rule_id,
        policy_action=ger.evaluated_policy_decision.decision.action.value,
        proof_digest=event.governed_proof.proof_digest,
        event_digest=accepted.receipt.host_attestation.payload_digest,
        key_id=accepted.receipt.host_attestation.key_id,
        receipt_path=receipt_rel,
        receipt_absolute_path=str(export_path.resolve()),
        verification_key_path=key_disp,
        verification_command=(
            f"uv run intergrax receipt verify {receipt_rel} --store {store_disp}"
        ),
        verification_valid=vr.valid,
        store_root=store_disp,
        provider_execution_succeeded=True,
        attestation_succeeded=True,
        state=accepted.state.value,
        recovery_command=None,
        create_calls=fake.create_calls,
        accept_calls=fake.accept_calls,
        cancel_calls=fake.cancel_calls,
    )
    (store_root / "demo_report.json").write_text(
        json.dumps(report.as_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return report
