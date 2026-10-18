# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Typed Tier-2 → host port for sanctioned external-work mutations (PROD-Q-R1-R7-Q)."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Mapping, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from intergrax.contracts.execution_identity import ExecutionId, RunId, TaskId
from intergrax.contracts.external_work import QuoteAcceptanceEvidence

META_SANCTIONED_MUTATION_OPERATION = "external_work.sanctioned_mutation_operation"
META_PRIOR_EXECUTION_ID = "external_work.prior_execution_id"


class SanctionedExternalWorkMutationOperation(StrEnum):
    CREATE = "create"
    ACCEPT_QUOTE = "accept_quote"
    CANCEL = "cancel"


class SanctionedExternalWorkCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: TaskId
    run_id: RunId
    principal_id: str = Field(min_length=1)
    tenant_id: str = Field(min_length=1)
    execution_id: ExecutionId | None = None
    metadata: Mapping[str, Any] = Field(default_factory=dict)
    message: str = ""


class SanctionedExternalWorkAcceptQuoteRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    execution_id: ExecutionId
    principal_id: str = Field(min_length=1)
    tenant_id: str = Field(min_length=1)
    acceptance: QuoteAcceptanceEvidence
    acceptance_idempotency_key: str = Field(min_length=1)
    metadata: Mapping[str, Any] = Field(default_factory=dict)


class SanctionedExternalWorkCancelRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    execution_id: ExecutionId
    principal_id: str = Field(min_length=1)
    tenant_id: str = Field(min_length=1)
    idempotency_key: str = Field(min_length=1)
    metadata: Mapping[str, Any] = Field(default_factory=dict)


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
    domain_summary: Mapping[str, Any] = Field(default_factory=dict)


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
    "SanctionedExternalWorkCancelRequest",
    "SanctionedExternalWorkCreateRequest",
    "SanctionedExternalWorkMutationOperation",
    "SanctionedExternalWorkOperationPort",
    "SanctionedExternalWorkOperationRequest",
    "SanctionedExternalWorkOperationResult",
]
