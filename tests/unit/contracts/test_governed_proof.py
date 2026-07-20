# © Artur Czarnecki. All rights reserved.

"""GEC-6 — GovernedProofProfile contract (descriptive only)."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from intergrax.contracts.governed_continuation import ContinuationReason
from intergrax.contracts.governed_proof import (
    EVIDENCE_KIND_QUOTE_ACCEPTANCE,
    GovernanceEvidenceRef,
    GovernedProofProfile,
    compose_governed_proof_profile,
    governance_evidence_ref_from_quote_acceptance,
)
from intergrax.contracts.runtime_policy import PolicyAction

_MODULE = Path("intergrax/contracts/governed_proof.py")


@pytest.mark.unit
@pytest.mark.gate
def test_compose_profile_preserves_identity_and_policy_refs() -> None:
    evidence = governance_evidence_ref_from_quote_acceptance(
        acceptance_id="acc-1",
        hitl_decision_id="hdec-1",
        policy_decision_ref="pol-1",
    )
    profile = compose_governed_proof_profile(
        principal_id="u1",
        tenant_id="tenant-a",
        task_id="task-1",
        run_id="run-1",
        action="ACCEPT_QUOTE",
        resource="sha256:" + ("ab" * 32),
        provider_id="provider-x",
        policy_action=PolicyAction.ALLOW,
        policy_rule_id="rule.allow",
        policy_reason="allowed",
        governance_evidence=evidence,
        continuation_reason=ContinuationReason.QUOTE,
        idempotency_key="idem-1",
        correlation_id="corr-1",
    )
    assert profile.schema_version == "governed_proof_profile.v1"
    assert profile.principal_id == "u1"
    assert profile.tenant_id == "tenant-a"
    assert profile.task_id == "task-1"
    assert profile.run_id == "run-1"
    assert profile.execution_ref == "run-1"
    assert profile.action == "ACCEPT_QUOTE"
    assert profile.provider_id == "provider-x"
    assert profile.policy_action is PolicyAction.ALLOW
    assert profile.policy_rule_id == "rule.allow"
    assert profile.idempotency_key == "idem-1"
    assert profile.correlation_id == "corr-1"
    assert profile.continuation_reason is ContinuationReason.QUOTE
    assert profile.governance_evidence is not None
    assert profile.governance_evidence.kind == EVIDENCE_KIND_QUOTE_ACCEPTANCE
    assert profile.governance_evidence.evidence_id == "acc-1"
    assert profile.governance_evidence.hitl_decision_id == "hdec-1"


@pytest.mark.unit
@pytest.mark.gate
def test_profile_forbids_transport_and_provider_payload_fields() -> None:
    dumped = compose_governed_proof_profile(
        principal_id="u1",
        task_id="t",
        run_id="r",
        action="CREATE_EXTERNAL_WORK",
        provider_id="p",
        policy_action=PolicyAction.ALLOW,
    ).model_dump()
    forbidden = {
        "http_headers",
        "headers",
        "request_body",
        "response_body",
        "json_rpc",
        "sdk_payload",
        "provider_payload",
        "transport",
        "signature",
        "receipt",
    }
    assert forbidden.isdisjoint(dumped.keys())
    with pytest.raises(ValidationError):
        GovernedProofProfile.model_validate(
            {
                "principal_id": "u1",
                "task_id": "t",
                "run_id": "r",
                "action": "CREATE_EXTERNAL_WORK",
                "provider_id": "p",
                "policy_action": PolicyAction.ALLOW,
                "http_headers": {"Authorization": "Bearer x"},
            }
        )


@pytest.mark.unit
@pytest.mark.gate
def test_evidence_ref_does_not_embed_acceptance_payload() -> None:
    ref = GovernanceEvidenceRef(
        kind=EVIDENCE_KIND_QUOTE_ACCEPTANCE,
        evidence_id="acc-2",
    )
    assert set(ref.model_dump()) == {
        "schema_version",
        "kind",
        "evidence_id",
        "hitl_decision_id",
        "interrupt_id",
        "policy_decision_ref",
    }
    assert "quote_id" not in ref.model_dump()
    assert "accepted_at" not in ref.model_dump()
    assert "actor" not in ref.model_dump()


@pytest.mark.unit
@pytest.mark.gate
def test_module_descriptive_only_no_tier_leaks() -> None:
    source = _MODULE.read_text(encoding="utf-8")
    lowered = source.lower()
    assert "descriptive" in lowered
    assert "receipt" in lowered  # documented as non-goal
    assert "ProofReceiptStore" not in source
    assert "hashlib" not in source
    assert "DocumentStore" not in source
    for needle in ("applications.", "agents.", "from applications", "from agents"):
        assert needle not in source
