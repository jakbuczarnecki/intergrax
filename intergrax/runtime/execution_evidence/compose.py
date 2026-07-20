# © Artur Czarnecki. All rights reserved.

"""Compose ExecutionBoundaryEvent and ProofReceipt after governed proof."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Mapping

from intergrax.contracts.execution_evidence.attestation import HostAttestor
from intergrax.contracts.execution_evidence.boundary_event import (
    ExecutionBoundaryEvent,
    GovernanceEvidenceSection,
    GovernedProofSection,
    PolicyDecisionSection,
    ProviderInvocationSection,
    SCHEMA_GOVERNED_EXECUTION_BOUNDARY_EVENT_V1,
)
from intergrax.contracts.execution_evidence.receipt import ProofReceipt
from intergrax.contracts.governed_proof import GovernedProofProfile
from intergrax.contracts.runtime_policy import PolicyAction, PolicyDecision
from intergrax.runtime.attestation.canonical_json import (
    canonical_json_bytes,
    stable_payload_hash,
)


@dataclass(frozen=True, slots=True)
class AttestationOutcome:
    """Host attestation result — distinguishes execution vs attestation failure."""

    execution_succeeded: bool
    attestation_succeeded: bool
    receipt: ProofReceipt | None
    event: ExecutionBoundaryEvent | None
    reason: str
    provider_invoked: bool = True


def proof_digest(proof: GovernedProofProfile) -> str:
    return stable_payload_hash(proof.model_dump(mode="json"))


def compose_execution_boundary_event(
    *,
    proof: GovernedProofProfile,
    policy_decision: PolicyDecision,
    provider_operation: str,
    invocation_id: str,
    invocation_completed_at: datetime,
    event_id: str | None = None,
    occurred_at: datetime | None = None,
    actor: str = "",
    require_policy_bundle: bool = True,
) -> ExecutionBoundaryEvent:
    """Host-owned composition — requires successful proof + ALLOW decision refs."""
    if policy_decision.action is not PolicyAction.ALLOW:
        raise ValueError("boundary_event_requires_allow_decision")
    if proof.policy_action is not PolicyAction.ALLOW:
        raise ValueError("boundary_event_requires_allow_proof")
    if require_policy_bundle and not policy_decision.has_attested_policy_bundle_refs():
        raise ValueError("policy_bundle_identity_missing")

    evidence_section: GovernanceEvidenceSection | None = None
    if proof.governance_evidence is not None:
        evidence_section = GovernanceEvidenceSection(
            kind=proof.governance_evidence.kind,
            evidence_id=proof.governance_evidence.evidence_id,
        )

    digest = proof_digest(proof)
    proof_id = (
        f"proof:{proof.task_id}:{proof.run_id}:{proof.action}:{digest.removeprefix('sha256:')[:16]}"
    )

    return ExecutionBoundaryEvent(
        event_id=event_id or f"ebe-{uuid.uuid4().hex}",
        occurred_at=occurred_at or datetime.now(timezone.utc),
        task_id=proof.task_id,
        run_id=proof.run_id,
        correlation_id=proof.correlation_id,
        idempotency_key=proof.idempotency_key,
        principal_id=proof.principal_id,
        tenant_id=proof.tenant_id,
        actor=actor,
        provider_id=proof.provider_id,
        action=proof.action,
        policy=PolicyDecisionSection(
            bundle_id=policy_decision.policy_bundle_id,
            bundle_version=policy_decision.policy_bundle_version,
            bundle_digest=policy_decision.policy_bundle_digest,
            rule_id=policy_decision.policy_rule_id,
            action=policy_decision.action,
            decision_id=policy_decision.decision_id,
            decision_ref=policy_decision.decision_id
            or policy_decision.policy_rule_id,
        ),
        governance_evidence=evidence_section,
        provider_invocation=ProviderInvocationSection(
            operation=provider_operation,
            invocation_id=invocation_id,
            outcome="success",
            completed_at=invocation_completed_at,
        ),
        governed_proof=GovernedProofSection(
            proof_id=proof_id,
            proof_digest=digest,
            proof=proof.model_dump(mode="json"),
        ),
    )


def produce_proof_receipt(
    *,
    event: ExecutionBoundaryEvent,
    attestor: HostAttestor,
    receipt_id: str | None = None,
) -> ProofReceipt:
    """Sign canonical event bytes and bind into a portable ProofReceipt."""
    payload = canonical_json_bytes(event.canonical_payload())
    attestation = attestor.attest(
        payload,
        schema=SCHEMA_GOVERNED_EXECUTION_BOUNDARY_EVENT_V1,
    )
    return ProofReceipt(
        receipt_id=receipt_id or f"rcpt-{uuid.uuid4().hex}",
        execution_boundary_event=event,
        host_attestation=attestation,
    )


def attest_after_governed_side_effect(
    *,
    proof: GovernedProofProfile | None,
    policy_decision: PolicyDecision | None,
    provider_operation: str,
    invocation_id: str,
    invocation_completed_at: datetime,
    attestor: HostAttestor | None,
    attestation_required: bool = True,
    execution_succeeded: bool,
    provider_invoked: bool,
    event_id: str | None = None,
    occurred_at: datetime | None = None,
    actor: str = "",
    receipt_id: str | None = None,
    clock: Callable[[], datetime] | None = None,
) -> AttestationOutcome:
    """Host orchestration after Tier-2 returns.

    Never retries provider execution. Distinguishes:
    - execution did not occur
    - execution succeeded but attestation failed
    """
    _ = clock
    if not execution_succeeded or not provider_invoked:
        return AttestationOutcome(
            execution_succeeded=False,
            attestation_succeeded=False,
            receipt=None,
            event=None,
            reason="execution_did_not_occur",
            provider_invoked=provider_invoked,
        )
    if proof is None:
        return AttestationOutcome(
            execution_succeeded=True,
            attestation_succeeded=False,
            receipt=None,
            event=None,
            reason="proof_missing_after_execution",
            provider_invoked=True,
        )
    if policy_decision is None or policy_decision.action is not PolicyAction.ALLOW:
        return AttestationOutcome(
            execution_succeeded=True,
            attestation_succeeded=False,
            receipt=None,
            event=None,
            reason="allow_decision_missing",
            provider_invoked=True,
        )
    if attestation_required and attestor is None:
        return AttestationOutcome(
            execution_succeeded=True,
            attestation_succeeded=False,
            receipt=None,
            event=None,
            reason="host_attestor_missing",
            provider_invoked=True,
        )
    try:
        event = compose_execution_boundary_event(
            proof=proof,
            policy_decision=policy_decision,
            provider_operation=provider_operation,
            invocation_id=invocation_id,
            invocation_completed_at=invocation_completed_at,
            event_id=event_id,
            occurred_at=occurred_at,
            actor=actor,
            require_policy_bundle=attestation_required,
        )
    except ValueError as exc:
        return AttestationOutcome(
            execution_succeeded=True,
            attestation_succeeded=False,
            receipt=None,
            event=None,
            reason=str(exc),
            provider_invoked=True,
        )
    if attestor is None:
        return AttestationOutcome(
            execution_succeeded=True,
            attestation_succeeded=False,
            receipt=None,
            event=event,
            reason="host_attestor_missing",
            provider_invoked=True,
        )
    try:
        receipt = produce_proof_receipt(
            event=event,
            attestor=attestor,
            receipt_id=receipt_id,
        )
    except Exception:  # noqa: BLE001 — never claim attested on signer failure
        return AttestationOutcome(
            execution_succeeded=True,
            attestation_succeeded=False,
            receipt=None,
            event=event,
            reason="attestation_failed",
            provider_invoked=True,
        )
    return AttestationOutcome(
        execution_succeeded=True,
        attestation_succeeded=True,
        receipt=receipt,
        event=event,
        reason="attested",
        provider_invoked=True,
    )


def invocation_id_from_adapter_metadata(
    metadata: Mapping[str, Any],
    *,
    fallback: str,
) -> str:
    for key in (
        "provider_invocation_id",
        "external_task_id",
        "invocation_id",
    ):
        raw = metadata.get(key)
        if isinstance(raw, str) and raw.strip():
            return raw.strip()
    return fallback
