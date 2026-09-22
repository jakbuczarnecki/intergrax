# © Artur Czarnecki. All rights reserved.

"""Execution-bound catalog tool invocation port (provider-neutral, UCA-6C-R5-R1)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from pydantic import BaseModel

from intergrax.contracts.tool_invocation_governance_approval_evidence import (
    ToolInvocationGovernanceApprovalEvidence,
)
from intergrax.tools.execution_models import ToolExecutionResult
from intergrax.tools.invocation_wiring import ToolInvocationWiringResolver


@dataclass(frozen=True, slots=True)
class ExecutionBoundCatalogToolInvokeRequest:
    """Typed catalog invocation under an active execution identity."""

    tool_id: str
    input: BaseModel
    tenant_id: str
    task_id: str
    run_id: str
    agent_id: str
    step_id: str
    correlation_request_id: str | None = None
    idempotency_key: str | None = None
    wiring_resolver: ToolInvocationWiringResolver | None = None
    governance_approval_evidence: ToolInvocationGovernanceApprovalEvidence | None = None


@runtime_checkable
class ExecutionBoundCatalogToolInvoker(Protocol):
    """Catalog tool gateway bound to tenant/run/task/caller — no RuntimeState exposure."""

    @property
    def caller_agent_id(self) -> str:
        """Default caller identity configured by host composition."""
        ...

    def invoke(
        self,
        request: ExecutionBoundCatalogToolInvokeRequest,
    ) -> ToolExecutionResult[BaseModel]: ...


__all__ = [
    "ExecutionBoundCatalogToolInvokeRequest",
    "ExecutionBoundCatalogToolInvoker",
]
