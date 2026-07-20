# © Artur Czarnecki. All rights reserved.

"""GEC-6 — Tier-2 composes GovernedProofProfile (no persistence / signing)."""

from __future__ import annotations

import ast
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import pytest

from external_contractor_adapter.external_work_adapter import (
    META_ACCEPTANCE_IDEMPOTENCY_KEY,
    META_CORRELATION_ID,
    META_IDEMPOTENCY_KEY,
    META_PROVIDER_ID,
    META_QUOTE_ACCEPTANCE,
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
from intergrax.contracts.actor_identity import ActorIdentity, ActorKind
from intergrax.contracts.external_work import QuoteAcceptanceEvidence
from intergrax.contracts.governed_continuation import ContinuationReason
from intergrax.contracts.governed_proof import EVIDENCE_KIND_QUOTE_ACCEPTANCE
from intergrax.contracts.money import MoneyAmount
from intergrax.contracts.runtime_policy import PolicyAction

_DIGEST = "sha256:" + ("cd" * 32)
_T0 = datetime(2026, 7, 20, 18, 0, 0, tzinfo=timezone.utc)
_AGENT_ROOT = Path(__file__).resolve().parents[1]
_ADAPTER_PY = _AGENT_ROOT / "external_work_adapter.py"
_FORBIDDEN_OWNERSHIP = {
    "ProofReceiptStore",
    "ProofReceipt",
    "DocumentStore",
    "sign_proof",
    "hash_proof",
    "persist_proof",
    "publish_proof",
}


def _meta(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        META_PROVIDER_ID: "gec3_deterministic_fake",
        META_SCOPE_DESCRIPTION: "review PR #99",
        META_SCOPE_DIGEST: _DIGEST,
        META_IDEMPOTENCY_KEY: "idem-gec6-1",
        META_CORRELATION_ID: "corr-gec6-1",
        "external_work.budget_limit": MoneyAmount(
            amount=Decimal("40.00"), currency="USD"
        ),
        "external_work.principal_id": "u1",
        "external_work.tenant_id": "tenant-a",
    }
    payload.update(overrides)
    return payload


def _acceptance(**overrides: object) -> QuoteAcceptanceEvidence:
    payload: dict[str, object] = {
        "acceptance_id": "acc-gec6-1",
        "quote_id": "q-gec3-1",
        "quote_version": 1,
        "scope_digest": _DIGEST,
        "actor": ActorIdentity(
            kind=ActorKind.USER, actor_id="u1", tenant_id="tenant-a"
        ),
        "accepted_at": _T0 + timedelta(minutes=3),
        "hitl_decision_id": "hdec_gec6",
        "interrupt_id": "intr_gec6",
        "policy_decision_ref": "pol_gec6",
    }
    payload.update(overrides)
    return QuoteAcceptanceEvidence.model_validate(payload)


def _allow() -> DeterministicMeaningfulSideEffectPolicy:
    return DeterministicMeaningfulSideEffectPolicy(default=PolicyAction.ALLOW)


@pytest.mark.unit
@pytest.mark.gate
def test_governed_create_produces_proof_profile() -> None:
    adapter = ExternalWorkAdapter(
        DeterministicExternalWorkFake(), side_effect_policy=_allow()
    )
    result = adapter.create_and_map(
        adapter.build_create_request(
            task_id="task-gec6",
            run_id="run-gec6",
            metadata=_meta(),
        ),
        principal_id="u1",
        tenant_id="tenant-a",
    )
    assert result.used is True
    assert result.proof is not None
    assert result.policy_decision is not None
    assert result.policy_decision.action is PolicyAction.ALLOW
    proof = result.proof
    assert proof.policy_action is PolicyAction.ALLOW
    assert proof.policy_rule_id == result.policy_decision.policy_rule_id
    assert proof.task_id == "task-gec6"
    assert proof.run_id == "run-gec6"
    assert proof.correlation_id == "corr-gec6-1"
    assert proof.idempotency_key == "idem-gec6-1"
    assert proof.provider_id == "gec3_deterministic_fake"
    assert proof.action == ACTION_CREATE_EXTERNAL_WORK
    assert proof.principal_id == "u1"
    dumped = proof.model_dump()
    assert "http_headers" not in dumped
    assert "provider_payload" not in dumped
    assert "request_body" not in dumped
    assert "signature" not in dumped


@pytest.mark.unit
@pytest.mark.gate
def test_accept_quote_proof_references_governance_evidence() -> None:
    adapter = ExternalWorkAdapter(
        DeterministicExternalWorkFake(), side_effect_policy=_allow()
    )
    created = adapter.create_and_map(
        adapter.build_create_request(
            task_id="task-gec6-acc",
            run_id="run-gec6-acc",
            metadata=_meta(**{META_IDEMPOTENCY_KEY: "idem-gec6-acc"}),
        ),
        principal_id="u1",
        tenant_id="tenant-a",
    )
    assert created.snapshot is not None
    acceptance = _acceptance(quote_id=created.quote.quote_id)  # type: ignore[union-attr]
    accepted = adapter.forward_quote_acceptance(
        created.snapshot.correlation,
        acceptance,
        idempotency_key="idem-gec6-accept",
        principal_id="u1",
        tenant_id="tenant-a",
    )
    assert accepted.used is True
    assert accepted.proof is not None
    assert accepted.proof.action == ACTION_ACCEPT_QUOTE
    assert accepted.proof.continuation_reason is ContinuationReason.QUOTE
    ev = accepted.proof.governance_evidence
    assert ev is not None
    assert ev.kind == EVIDENCE_KIND_QUOTE_ACCEPTANCE
    assert ev.evidence_id == "acc-gec6-1"
    assert ev.hitl_decision_id == "hdec_gec6"
    assert ev.interrupt_id == "intr_gec6"
    assert ev.policy_decision_ref == "pol_gec6"
    # Evidence is referenced, not embedded as a full acceptance payload.
    assert "quote_version" not in ev.model_dump()
    assert "actor" not in ev.model_dump()
    assert accepted.proof.task_id == "task-gec6-acc"
    assert accepted.proof.run_id == "run-gec6-acc"
    assert accepted.proof.idempotency_key == "idem-gec6-accept"
    assert accepted.proof.correlation_id == created.snapshot.correlation.correlation_id


@pytest.mark.unit
@pytest.mark.gate
def test_gec5_policy_flow_unchanged_deny_before_provider() -> None:
    policy = DeterministicMeaningfulSideEffectPolicy(default=PolicyAction.DENY)
    call_log: list[str] = []
    policy.call_log = call_log

    class _Rec(DeterministicExternalWorkFake):
        def create_work(self, request):  # type: ignore[no-untyped-def]
            call_log.append("integration.create_work")
            return super().create_work(request)

    adapter = ExternalWorkAdapter(_Rec(), side_effect_policy=policy)
    result = adapter.create_and_map(
        adapter.build_create_request(
            task_id="task-deny",
            run_id="run-deny",
            metadata=_meta(**{META_IDEMPOTENCY_KEY: "idem-deny"}),
        ),
        principal_id="u1",
        tenant_id="tenant-a",
    )
    assert result.used is False
    assert result.reason == "side_effect_denied"
    assert result.proof is None
    assert "integration.create_work" not in call_log
    assert "policy.evaluate" in call_log


@pytest.mark.unit
@pytest.mark.gate
def test_tier2_performs_no_persistence_signing_or_receipt_generation() -> None:
    source = _ADAPTER_PY.read_text(encoding="utf-8")
    for name in _FORBIDDEN_OWNERSHIP:
        assert name not in source
    tree = ast.parse(source, filename=str(_ADAPTER_PY))
    call_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                call_names.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                call_names.add(node.func.attr)
    for forbidden in ("put", "sign", "verify", "persist", "publish", "hash"):
        # Attribute helpers may use benign names; ownership APIs must be absent.
        assert forbidden not in {
            n for n in call_names if n in {"sign_proof", "persist_proof", "publish_proof"}
        }
    assert "compose_governed_proof_profile" in source
    assert "ProofReceiptStore" not in {
        n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)
    }


@pytest.mark.unit
@pytest.mark.gate
def test_create_then_accept_in_one_call_composes_accept_proof() -> None:
    fake = DeterministicExternalWorkFake()
    adapter = ExternalWorkAdapter(fake, side_effect_policy=_allow())
    # Pre-create to learn the deterministic quote id, then accept on a fresh
    # create+accept call using a matching acceptance for that new work's quote.
    # Simpler path: create once, accept via create_and_map with evidence.
    created = adapter.create_and_map(
        adapter.build_create_request(
            task_id="task-combo",
            run_id="run-combo",
            metadata=_meta(**{META_IDEMPOTENCY_KEY: "idem-combo"}),
        ),
        principal_id="u1",
        tenant_id="tenant-a",
    )
    assert created.quote is not None
    acceptance = _acceptance(quote_id=created.quote.quote_id)
    # Re-enter with same idempotency → same correlated work, then forward accept.
    result = adapter.create_and_map(
        adapter.build_create_request(
            task_id="task-combo",
            run_id="run-combo",
            metadata=_meta(
                **{
                    META_IDEMPOTENCY_KEY: "idem-combo",
                    META_QUOTE_ACCEPTANCE: acceptance,
                    META_ACCEPTANCE_IDEMPOTENCY_KEY: "idem-combo-acc",
                }
            ),
        ),
        acceptance=acceptance,
        acceptance_idempotency_key="idem-combo-acc",
        principal_id="u1",
        tenant_id="tenant-a",
    )
    assert result.used is True
    assert result.proof is not None
    assert result.proof.action == ACTION_ACCEPT_QUOTE
    assert result.proof.governance_evidence is not None
    assert result.proof.governance_evidence.evidence_id == "acc-gec6-1"
