# © Artur Czarnecki. All rights reserved.

"""Host-owned governed execution boundary event (ADR-EXECUTION-BOUNDARY-EVENT-001).

Schema id: ``governed_execution_boundary_event.v1`` — distinct from harness
``execution_boundary_event.v1`` (tool/step BoundaryAttest export).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from intergrax.contracts.runtime_policy import PolicyAction

SCHEMA_GOVERNED_EXECUTION_BOUNDARY_EVENT_V1: Final = (
    "governed_execution_boundary_event.v1"
)
_NON_EMPTY = Field(min_length=1)


class PolicyDecisionSection(BaseModel):
    """Exact policy decision / pack used before the side effect."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    bundle_id: str = _NON_EMPTY
    bundle_version: str = _NON_EMPTY
    bundle_digest: str = _NON_EMPTY
    rule_id: str = ""
    action: PolicyAction
    decision_id: str = ""
    decision_ref: str = ""

    @field_validator("bundle_id", "bundle_version", "bundle_digest")
    @classmethod
    def _strip_required(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("must be non-empty")
        return normalized


class GovernanceEvidenceSection(BaseModel):
    """Optional governance evidence pointer (never embeds payload)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: str = _NON_EMPTY
    evidence_id: str = _NON_EMPTY

    @field_validator("kind", "evidence_id")
    @classmethod
    def _strip_required(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("must be non-empty")
        return normalized


class ProviderInvocationSection(BaseModel):
    """Provider-bound invocation reference — no transport response bodies."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    operation: str = _NON_EMPTY
    invocation_id: str = _NON_EMPTY
    outcome: Literal["success", "failure"] = "success"
    completed_at: datetime

    @field_validator("operation", "invocation_id")
    @classmethod
    def _strip_required(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("must be non-empty")
        return normalized


class GovernedProofSection(BaseModel):
    """Link to the descriptive proof composed after the side effect."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    proof_id: str = _NON_EMPTY
    proof_digest: str = _NON_EMPTY
    # Optional embedded canonical proof representation (no secrets).
    proof: dict[str, Any] | None = None

    @field_validator("proof_id", "proof_digest")
    @classmethod
    def _strip_required(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("must be non-empty")
        return normalized


class ExecutionBoundaryEvent(BaseModel):
    """Completed governed execution boundary — never authorizes or resumes."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_id: Literal["governed_execution_boundary_event.v1"] = (
        SCHEMA_GOVERNED_EXECUTION_BOUNDARY_EVENT_V1
    )
    event_id: str = _NON_EMPTY
    occurred_at: datetime

    task_id: str = _NON_EMPTY
    run_id: str = _NON_EMPTY
    correlation_id: str | None = None
    idempotency_key: str | None = None

    principal_id: str = _NON_EMPTY
    tenant_id: str | None = None
    actor: str = ""

    provider_id: str = _NON_EMPTY
    action: str = _NON_EMPTY

    policy: PolicyDecisionSection
    governance_evidence: GovernanceEvidenceSection | None = None
    provider_invocation: ProviderInvocationSection
    governed_proof: GovernedProofSection

    @field_validator(
        "event_id",
        "task_id",
        "run_id",
        "principal_id",
        "provider_id",
        "action",
    )
    @classmethod
    def _strip_required(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("must be non-empty")
        return normalized

    def canonical_payload(self) -> dict[str, Any]:
        """JSON-mode dump suitable for canonical serialization / digest."""
        return self.model_dump(mode="json")
