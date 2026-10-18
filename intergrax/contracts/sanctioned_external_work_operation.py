# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Typed Tier-2 → host port for sanctioned external-work mutations (PROD-Q-R1-R7-Q)."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from intergrax.contracts.execution_identity import ExecutionId, RunId, TaskId
from intergrax.contracts.external_work import QuoteAcceptanceEvidence

META_SANCTIONED_MUTATION_OPERATION = "external_work.sanctioned_mutation_operation"
META_PRIOR_EXECUTION_ID = "external_work.prior_execution_id"
META_ADAPTER_PROVIDER_ID = "external_work.provider_id"
META_ADAPTER_SCOPE_DESCRIPTION = "external_work.scope_description"
META_ADAPTER_SCOPE_DIGEST = "external_work.scope_digest"
META_ADAPTER_IDEMPOTENCY_KEY = "external_work.idempotency_key"
META_ADAPTER_CORRELATION_ID = "external_work.correlation_id"
META_ADAPTER_WORKSPACE_REF = "external_work.workspace_ref"
META_ADAPTER_BUDGET_LIMIT = "external_work.budget_limit"
META_ADAPTER_REQUESTED_CAPABILITY = "external_work.requested_capability"
META_ADAPTER_SKIP_ENRICHMENT = "external_work.skip_enrichment"


class SanctionedExternalWorkMutationOperation(StrEnum):
    CREATE = "create"
    ACCEPT_QUOTE = "accept_quote"
    CANCEL = "cancel"


class SanctionedExternalWorkAdapterInvocationMetadata(BaseModel):
    """Typed adapter metadata passed host → orchestrator (parsed in Tier-2)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    provider_id: str | None = None
    scope_description: str | None = None
    scope_digest: str | None = None
    idempotency_key: str | None = None
    correlation_id: str | None = None
    workspace_ref: str | None = None
    budget_limit: str | None = None
    requested_capability: str | None = None
    skip_enrichment: bool | None = None

    def to_adapter_metadata(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        if self.provider_id:
            out[META_ADAPTER_PROVIDER_ID] = self.provider_id
        if self.scope_description:
            out[META_ADAPTER_SCOPE_DESCRIPTION] = self.scope_description
        if self.scope_digest:
            out[META_ADAPTER_SCOPE_DIGEST] = self.scope_digest
        if self.idempotency_key:
            out[META_ADAPTER_IDEMPOTENCY_KEY] = self.idempotency_key
        if self.correlation_id:
            out[META_ADAPTER_CORRELATION_ID] = self.correlation_id
        if self.workspace_ref:
            out[META_ADAPTER_WORKSPACE_REF] = self.workspace_ref
        if self.budget_limit:
            out[META_ADAPTER_BUDGET_LIMIT] = self.budget_limit
        if self.requested_capability:
            out[META_ADAPTER_REQUESTED_CAPABILITY] = self.requested_capability
        if self.skip_enrichment is not None:
            out[META_ADAPTER_SKIP_ENRICHMENT] = self.skip_enrichment
        return out


class SanctionedExternalWorkDomainSummary(BaseModel):
    """Structured domain summary for reflex step output (no open metadata map)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    used: bool
    reason: str = Field(min_length=1)
    capability: str | None = None
    status: str | None = None
    sanctioned_operation: str | None = None
    host_state: str | None = None
    execution_id: str | None = None
    provider_invocation_id: str | None = None
    correlation: dict[str, Any] | None = None
    snapshot: dict[str, Any] | None = None
    quote: dict[str, Any] | None = None
    error_code: str | None = None
    error_message: str | None = None
    error_retryable: bool | None = None

    @classmethod
    def from_adapter_summary(
        cls,
        summary: dict[str, object],
        *,
        sanctioned_operation: str,
        host_state: str,
        execution_id: str | None,
        provider_invocation_id: str | None,
    ) -> SanctionedExternalWorkDomainSummary:
        return cls(
            used=bool(summary.get("used")),
            reason=str(summary.get("reason") or "unknown"),
            capability=(
                str(summary["capability"]) if summary.get("capability") is not None else None
            ),
            status=str(summary["status"]) if summary.get("status") is not None else None,
            sanctioned_operation=sanctioned_operation,
            host_state=host_state,
            execution_id=execution_id,
            provider_invocation_id=provider_invocation_id,
            correlation=(
                dict(summary["correlation"])
                if isinstance(summary.get("correlation"), dict)
                else None
            ),
            snapshot=(
                dict(summary["snapshot"])
                if isinstance(summary.get("snapshot"), dict)
                else None
            ),
            quote=dict(summary["quote"]) if isinstance(summary.get("quote"), dict) else None,
            error_code=(
                str(summary["error_code"]) if summary.get("error_code") is not None else None
            ),
            error_message=(
                str(summary["error_message"])
                if summary.get("error_message") is not None
                else None
            ),
            error_retryable=(
                bool(summary["error_retryable"])
                if summary.get("error_retryable") is not None
                else None
            ),
        )

    def to_step_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "used": self.used,
            "reason": self.reason,
            "sanctioned_operation": self.sanctioned_operation,
            "host_state": self.host_state,
        }
        if self.capability is not None:
            payload["capability"] = self.capability
        if self.status is not None:
            payload["status"] = self.status
        if self.execution_id is not None:
            payload["execution_id"] = self.execution_id
        if self.provider_invocation_id is not None:
            payload["provider_invocation_id"] = self.provider_invocation_id
        if self.correlation is not None:
            payload["correlation"] = self.correlation
        if self.snapshot is not None:
            payload["snapshot"] = self.snapshot
        if self.quote is not None:
            payload["quote"] = self.quote
        if self.error_code is not None:
            payload["error_code"] = self.error_code
        if self.error_message is not None:
            payload["error_message"] = self.error_message
        if self.error_retryable is not None:
            payload["error_retryable"] = self.error_retryable
        return payload


class SanctionedExternalWorkCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: TaskId
    run_id: RunId
    principal_id: str = Field(min_length=1)
    tenant_id: str = Field(min_length=1)
    execution_id: ExecutionId | None = None
    adapter_metadata: SanctionedExternalWorkAdapterInvocationMetadata = Field(
        default_factory=SanctionedExternalWorkAdapterInvocationMetadata,
    )
    message: str = ""


class SanctionedExternalWorkAcceptQuoteRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    execution_id: ExecutionId
    principal_id: str = Field(min_length=1)
    tenant_id: str = Field(min_length=1)
    acceptance: QuoteAcceptanceEvidence
    acceptance_idempotency_key: str = Field(min_length=1)
    adapter_metadata: SanctionedExternalWorkAdapterInvocationMetadata = Field(
        default_factory=SanctionedExternalWorkAdapterInvocationMetadata,
    )


class SanctionedExternalWorkCancelRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    execution_id: ExecutionId
    principal_id: str = Field(min_length=1)
    tenant_id: str = Field(min_length=1)
    idempotency_key: str = Field(min_length=1)
    adapter_metadata: SanctionedExternalWorkAdapterInvocationMetadata = Field(
        default_factory=SanctionedExternalWorkAdapterInvocationMetadata,
    )


class SanctionedExternalWorkOperationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    operation: SanctionedExternalWorkMutationOperation
    create: SanctionedExternalWorkCreateRequest | None = None
    accept_quote: SanctionedExternalWorkAcceptQuoteRequest | None = None
    cancel: SanctionedExternalWorkCancelRequest | None = None


class SanctionedExternalWorkOperationResult(BaseModel):
    """Host-owned outcome — adapter view plus correlation for reflex domain summary."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    admitted: bool
    reason: str = Field(min_length=1)
    execution_id: str | None = None
    provider_invocation_id: str | None = None
    host_state: str | None = None
    adapter_used: bool = False
    adapter_reason: str | None = None
    provider_mutation_dispatched: bool = False
    domain_summary: SanctionedExternalWorkDomainSummary


@runtime_checkable
class SanctionedExternalWorkOperationPort(Protocol):
    """Sanctioned external-work mutations — orchestrator-backed, no store exposure."""

    def execute(
        self,
        request: SanctionedExternalWorkOperationRequest,
    ) -> SanctionedExternalWorkOperationResult: ...


__all__ = [
    "META_PRIOR_EXECUTION_ID",
    "META_SANCTIONED_MUTATION_OPERATION",
    "SanctionedExternalWorkAcceptQuoteRequest",
    "SanctionedExternalWorkAdapterInvocationMetadata",
    "SanctionedExternalWorkCancelRequest",
    "SanctionedExternalWorkCreateRequest",
    "SanctionedExternalWorkDomainSummary",
    "SanctionedExternalWorkMutationOperation",
    "SanctionedExternalWorkOperationPort",
    "SanctionedExternalWorkOperationRequest",
    "SanctionedExternalWorkOperationResult",
]
