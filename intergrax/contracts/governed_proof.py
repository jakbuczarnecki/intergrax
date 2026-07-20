# © Artur Czarnecki. All rights reserved.

"""Governed proof profile — descriptive composition over existing governance facts (GEC-6).

A proof profile answers *what must be provable* after a governed external side
effect. It is **not** a receipt, audit log, authorization mechanism, or
persistence record.

```text
Who?  principal (+ tenant)
Where? Nexus task_id / run_id
What? action + resource + provider
Why allowed? PolicyAction / policy rule refs (not recomputed)
What evidence? governance evidence references (not embedded payloads)
How correlated? correlation_id + idempotency_key
```

Reuses ``PolicyAction``, ``ContinuationReason``, and string refs into existing
artifacts (e.g. ``QuoteAcceptanceEvidence.acceptance_id``). Does not evaluate
policy, resume Nexus, sign, hash, store, or publish.
"""

from __future__ import annotations

from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from intergrax.contracts.governed_continuation import ContinuationReason
from intergrax.contracts.runtime_policy import PolicyAction

SCHEMA_GOVERNED_PROOF_PROFILE_V1: Final = "governed_proof_profile.v1"
SCHEMA_GOVERNANCE_EVIDENCE_REF_V1: Final = "governance_evidence_ref.v1"

_NON_EMPTY = Field(min_length=1)


class GovernanceEvidenceRef(BaseModel):
    """Pointer to an existing governance artifact — never embeds the payload."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["governance_evidence_ref.v1"] = (
        SCHEMA_GOVERNANCE_EVIDENCE_REF_V1
    )
    kind: str = _NON_EMPTY
    evidence_id: str = _NON_EMPTY
    hitl_decision_id: str | None = None
    interrupt_id: str | None = None
    policy_decision_ref: str | None = None

    @field_validator("kind", "evidence_id")
    @classmethod
    def _strip_required(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("must be non-empty")
        return normalized

    @field_validator("hitl_decision_id", "interrupt_id", "policy_decision_ref")
    @classmethod
    def _strip_optional(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None


class GovernedProofProfile(BaseModel):
    """Minimum descriptive facts for a governed external side effect.

    Immutable. Provider-/transport-neutral. Does not authorize or persist.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["governed_proof_profile.v1"] = (
        SCHEMA_GOVERNED_PROOF_PROFILE_V1
    )
    principal_id: str = _NON_EMPTY
    tenant_id: str | None = None
    task_id: str = _NON_EMPTY
    run_id: str = _NON_EMPTY
    action: str = _NON_EMPTY
    resource: str | None = None
    provider_id: str = _NON_EMPTY
    policy_action: PolicyAction
    policy_rule_id: str = ""
    policy_reason: str = ""
    governance_evidence: GovernanceEvidenceRef | None = None
    continuation_reason: ContinuationReason | None = None
    idempotency_key: str | None = None
    correlation_id: str | None = None
    execution_ref: str | None = None

    @field_validator("principal_id", "task_id", "run_id", "action", "provider_id")
    @classmethod
    def _strip_required(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("must be non-empty")
        return normalized

    @field_validator(
        "tenant_id",
        "resource",
        "idempotency_key",
        "correlation_id",
        "execution_ref",
    )
    @classmethod
    def _strip_optional(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None


# Well-known evidence kinds (consumers may add their own strings).
EVIDENCE_KIND_QUOTE_ACCEPTANCE: Final = "quote_acceptance_evidence"


def compose_governed_proof_profile(
    *,
    principal_id: str,
    task_id: str,
    run_id: str,
    action: str,
    provider_id: str,
    policy_action: PolicyAction,
    tenant_id: str | None = None,
    resource: str | None = None,
    policy_rule_id: str = "",
    policy_reason: str = "",
    governance_evidence: GovernanceEvidenceRef | None = None,
    continuation_reason: ContinuationReason | None = None,
    idempotency_key: str | None = None,
    correlation_id: str | None = None,
    execution_ref: str | None = None,
) -> GovernedProofProfile:
    """Build a proof profile from already-known platform facts.

    Does not evaluate policy, mint identifiers, or touch storage.
    """
    return GovernedProofProfile(
        principal_id=principal_id,
        tenant_id=tenant_id,
        task_id=task_id,
        run_id=run_id,
        action=action,
        resource=resource,
        provider_id=provider_id,
        policy_action=policy_action,
        policy_rule_id=policy_rule_id,
        policy_reason=policy_reason,
        governance_evidence=governance_evidence,
        continuation_reason=continuation_reason,
        idempotency_key=idempotency_key,
        correlation_id=correlation_id,
        execution_ref=execution_ref if execution_ref is not None else run_id,
    )


def governance_evidence_ref_from_quote_acceptance(
    *,
    acceptance_id: str,
    hitl_decision_id: str | None = None,
    interrupt_id: str | None = None,
    policy_decision_ref: str | None = None,
) -> GovernanceEvidenceRef:
    """QUOTE specialization: reference acceptance without copying the payload."""
    return GovernanceEvidenceRef(
        kind=EVIDENCE_KIND_QUOTE_ACCEPTANCE,
        evidence_id=acceptance_id,
        hitl_decision_id=hitl_decision_id,
        interrupt_id=interrupt_id,
        policy_decision_ref=policy_decision_ref,
    )
