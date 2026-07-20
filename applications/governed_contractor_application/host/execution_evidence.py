# © Artur Czarnecki. All rights reserved.

"""Host orchestration: GovernedProofProfile → EBE → HostAttestation → ProofReceipt.

Tier-2 owns provider execution + proof composition. This module never retries
provider side effects when attestation fails.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Mapping

from external_contractor_adapter.schemas.adapt_result import ExternalWorkAdapterResult
from external_contractor_adapter.side_effect_actions import (
    ACTION_ACCEPT_QUOTE,
    ACTION_CANCEL_EXTERNAL_WORK,
    ACTION_CREATE_EXTERNAL_WORK,
)
from intergrax.contracts.execution_evidence.attestation import HostAttestor
from intergrax.contracts.runtime_policy import PolicyAction
from intergrax.runtime.execution_evidence.compose import (
    AttestationOutcome,
    attest_after_governed_side_effect,
    invocation_id_from_adapter_metadata,
)

_ACTION_TO_OPERATION: Mapping[str, str] = {
    ACTION_CREATE_EXTERNAL_WORK: "create_work",
    ACTION_ACCEPT_QUOTE: "submit_quote_acceptance",
    ACTION_CANCEL_EXTERNAL_WORK: "cancel_work",
}


def produce_attested_receipt_for_adapter_result(
    result: ExternalWorkAdapterResult,
    *,
    attestor: HostAttestor | None,
    attestation_required: bool = True,
    actor: str = "governed_contractor_host",
    event_id: str | None = None,
    receipt_id: str | None = None,
    occurred_at: datetime | None = None,
) -> AttestationOutcome:
    """Compose and sign a portable receipt after a successful governed side effect.

    Failure semantics:
    - policy DENY / unused result → ``execution_did_not_occur`` (no receipt)
    - provider/proof failure → no success receipt
    - attestation failure after success → ``execution_succeeded=True``,
      ``attestation_succeeded=False`` (never retries provider)
    """
    provider_invoked = _provider_was_invoked(result)
    execution_succeeded = bool(
        result.used
        and result.proof is not None
        and result.policy_decision is not None
        and result.policy_decision.action is PolicyAction.ALLOW
        and provider_invoked
    )
    operation = _resolve_operation(result)
    invocation_id = invocation_id_from_adapter_metadata(
        dict(result.metadata),
        fallback=_fallback_invocation_id(result),
    )
    completed_at = occurred_at or datetime.now(timezone.utc)
    return attest_after_governed_side_effect(
        proof=result.proof,
        policy_decision=result.policy_decision,
        provider_operation=operation,
        invocation_id=invocation_id,
        invocation_completed_at=completed_at,
        attestor=attestor,
        attestation_required=attestation_required,
        execution_succeeded=execution_succeeded,
        provider_invoked=provider_invoked,
        event_id=event_id,
        occurred_at=occurred_at,
        actor=actor,
        receipt_id=receipt_id,
    )


def _provider_was_invoked(result: ExternalWorkAdapterResult) -> bool:
    reason = result.reason or ""
    if reason in {
        "policy_denied",
        "side_effect_identity_missing",
        "side_effect_policy_missing",
        "side_effect_principal_missing",
        "external_work_integration_missing",
    }:
        return False
    if result.policy_decision is not None and result.policy_decision.action is not PolicyAction.ALLOW:
        return False
    if not result.used:
        return False
    # Successful CREATE/ACCEPT/CANCEL paths expose snapshot and proof.
    return result.proof is not None or result.snapshot is not None


def _resolve_operation(result: ExternalWorkAdapterResult) -> str:
    if result.proof is not None and result.proof.action in _ACTION_TO_OPERATION:
        return _ACTION_TO_OPERATION[result.proof.action]
    meta_op = result.metadata.get("provider_operation")
    if isinstance(meta_op, str) and meta_op.strip():
        return meta_op.strip()
    return "external_work_side_effect"


def _fallback_invocation_id(result: ExternalWorkAdapterResult) -> str:
    if result.snapshot is not None:
        corr = result.snapshot.correlation
        external_id = getattr(corr, "external_task_id", None)
        if isinstance(external_id, str) and external_id.strip():
            return external_id.strip()
        if result.proof and result.proof.idempotency_key:
            return f"{result.proof.action}:{result.proof.idempotency_key}"
    if result.proof is not None:
        return f"{result.proof.task_id}:{result.proof.run_id}:{result.proof.action}"
    return "invocation:unknown"
