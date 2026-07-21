# © Artur Czarnecki. All rights reserved.

"""Reproducible offline governed-contractor demo (PC-8).

No network. Deterministic fake provider + local Ed25519 attestor.
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
from intergrax.runtime.execution_evidence.verify import (
    StaticKeyResolver,
    verify_proof_receipt,
)
from intergrax.runtime.policy.runtime_policy_bundle_evaluator import (
    RuntimePolicyBundleEvaluator,
)
from governed_contractor_application.host.orchestrator import (
    GovernedExternalWorkOrchestrator,
)
from governed_contractor_application.host.stores import FilesystemHostStore

_DIGEST = "sha256:" + ("ab" * 32)
_T0 = datetime(2026, 7, 21, 8, 0, 0, tzinfo=timezone.utc)
_PROVIDER_ID = "gec3_deterministic_fake"


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
    verification_valid: bool
    store_root: str

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
            "verification_valid": self.verification_valid,
            "store_root": self.store_root,
        }


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


def run_offline_governed_contractor_demo(
    *,
    store_root: Path,
    task_id: str = "task-offline-demo",
    run_id: str = "run-offline-demo",
) -> OfflineDemoReport:
    """Full CREATE → quote → human fixture → ACCEPT → attested receipt → verify."""
    store_root = Path(store_root)
    store_root.mkdir(parents=True, exist_ok=True)
    store = FilesystemHostStore(store_root)
    bundle = build_demo_policy_bundle()
    policy = RuntimePolicyBundleEvaluator(bundle, clock=lambda: _T0)
    fake = DeterministicExternalWorkFake()
    adapter = ExternalWorkAdapter(fake, side_effect_policy=policy)
    attestor = build_deterministic_test_attestor(
        key_id="governed-contractor-offline-demo-1",
        clock=lambda: _T0,
        attestation_id="att-offline-demo",
    )
    orch = GovernedExternalWorkOrchestrator(
        adapter=adapter,
        policy=policy,
        bundle=bundle,
        attestor=attestor,
        capabilities=quote_first_partner_capability_fixture(
            provider_id=_PROVIDER_ID,
        ),
        execution_store=store,
        receipt_store=store,
        bundle_store=store,
        continuation_store=store,
        clock=lambda: _T0,
    )
    meta = {
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
    created = orch.create(
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
    surfaced = orch.surface_continuation(
        execution_id=created.execution_id or "exec-offline-create",
        adapter_result=created.adapter_result,
        run_id=run_id,
    )
    _ = surfaced
    acceptance = QuoteAcceptanceEvidence.model_validate(
        {
            "acceptance_id": "acc-offline-demo",
            "quote_id": created.adapter_result.quote.quote_id,  # type: ignore[union-attr]
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
    # ACCEPT uses a distinct execution_id (separate side effect / invocation).
    accepted = orch.accept(
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
    if accepted.receipt is None or accepted.governed_result is None:
        raise RuntimeError(f"accept_attestation_failed:{accepted.reason}")
    receipt_path = store_root / "receipts" / "exec-offline-accept.json"
    export_path = store_root / "export" / "accept_receipt.json"
    export_path.parent.mkdir(parents=True, exist_ok=True)
    export_path.write_text(
        accepted.receipt.model_dump_json(indent=2), encoding="utf-8"
    )
    resolver = StaticKeyResolver(
        {attestor.key_id: attestor.public_key_bytes},
        current_key_id=attestor.key_id,
    )
    vr = verify_proof_receipt(
        accepted.receipt,
        key_resolver=resolver,
        require_policy_bundle_artifact=True,
    )
    ger = accepted.governed_result
    event = accepted.receipt.execution_boundary_event
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
        receipt_path=str(export_path.resolve()),
        verification_valid=vr.valid,
        store_root=str(store_root.resolve()),
    )
    (store_root / "demo_report.json").write_text(
        json.dumps(report.as_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _ = receipt_path
    return report
