# © Artur Czarnecki. All rights reserved.

"""Host orchestration: GovernedProofProfile → EBE → HostAttestation → ProofReceipt.

Preferred production path: ``GovernedExecutionResult`` via
``attest_governed_execution_result`` (orchestrator). This module retains the
legacy adapter-result composer for compatibility demos.

Tier-2 owns provider execution + proof composition. This module never retries
provider side effects when attestation fails. Heuristic invocation fallback is
compatibility-only — strict attested paths require first-class
``ProviderInvocation.invocation_id``.
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
from intergrax.contracts.governed_execution_result import GovernedExecutionResult
from intergrax.contracts.runtime_policy_bundle import ImmutableRuntimePolicyBundle
from intergrax.runtime.execution_evidence.compose import (
    AttestationOutcome,
    attest_after_governed_side_effect,
    attest_governed_execution_result,
    invocation_id_from_adapter_metadata,
)

_ACTION_TO_OPERATION: Mapping[str, str] = {
    ACTION_CREATE_EXTERNAL_WORK: "create_work",
    ACTION_ACCEPT_QUOTE: "submit_quote_acceptance",
    ACTION_CANCEL_EXTERNAL_WORK: "cancel_work",
}


def produce_attested_receipt_for_governed_result(
    result: GovernedExecutionResult,
    *,
    attestor: HostAttestor | None,
    policy_bundle_artifact: ImmutableRuntimePolicyBundle | None = None,
    attestation_required: bool = True,
    actor: str = "governed_contractor_host",
    event_id: str | None = None,
    receipt_id: str | None = None,
    occurred_at: datetime | None = None,
) -> AttestationOutcome:
    """Preferred host path — atomic GER, strict first-class invocation."""
    return attest_governed_execution_result(
        result,
        attestor=attestor,
        policy_bundle_artifact=policy_bundle_artifact,
        attestation_required=attestation_required,
        actor=actor,
        event_id=event_id,
        receipt_id=receipt_id,
        occurred_at=occurred_at,
        require_first_class_invocation=True,
    )


def produce_attested_receipt_for_adapter_result(
    result: ExternalWorkAdapterResult,
    *,
    attestor: HostAttestor | None,
    attestation_required: bool = True,
    actor: str = "governed_contractor_host",
    event_id: str | None = None,
    receipt_id: str | None = None,
    occurred_at: datetime | None = None,
    allow_invocation_fallback: bool = True,
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
    # Prefer explicit per-invocation ids only. Bare ``external_task_id`` is a
    # task correlation key shared across CREATE/ACCEPT/CANCEL — not unique.
    fallback = (
        _fallback_invocation_id(result)
        if allow_invocation_fallback
        else "invocation:unknown"
    )
    invocation_id = invocation_id_from_adapter_metadata(
        dict(result.metadata),
        fallback=fallback,
        prefer_keys=("provider_invocation_id", "invocation_id"),
    )
    if (
        attestation_required
        and not allow_invocation_fallback
        and invocation_id == "invocation:unknown"
    ):
        return AttestationOutcome(
            execution_succeeded=execution_succeeded,
            attestation_succeeded=False,
            receipt=None,
            event=None,
            reason="first_class_invocation_id_required",
            provider_invoked=provider_invoked,
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
        "side_effect_authorization_boundary_missing",
        "side_effect_tenant_missing",
        "side_effect_workspace_missing",
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
    """Stable invocation reference when adapter metadata lacks provider_invocation_id.

    Prefer ``action:external_task_id:idempotency_key`` so CREATE / ACCEPT / CANCEL
    on the same external task remain distinct. Bare ``external_task_id`` alone is
    insufficient as a per-invocation identity.
    """
    action = result.proof.action if result.proof is not None else ""
    idem = ""
    if result.proof is not None and result.proof.idempotency_key:
        idem = result.proof.idempotency_key.strip()
    external_id = ""
    if result.snapshot is not None:
        corr = result.snapshot.correlation
        raw = getattr(corr, "external_task_id", None)
        if isinstance(raw, str) and raw.strip():
            external_id = raw.strip()
    if action and external_id and idem:
        return f"{action}:{external_id}:{idem}"
    if action and idem:
        return f"{action}:{idem}"
    if action and external_id:
        return f"{action}:{external_id}"
    if external_id:
        return external_id
    if result.proof is not None:
        return f"{result.proof.task_id}:{result.proof.run_id}:{result.proof.action}"
    return "invocation:unknown"
