# © Artur Czarnecki. All rights reserved.

"""Typed catalog tool suspended-operation payload (UCA-6C-R6)."""

from __future__ import annotations

from pydantic import ConfigDict, Field

from intergrax.contracts.execution.suspended_operation.codec import SuspendedOperationPayload
from intergrax.tools.providers.sandbox.contracts import CodeExecInput

CODE_EXEC_INPUT_SCHEMA_ID = "sandbox.code_exec_input.v1"
EXECUTION_BOUND_CATALOG_TOOL_PAYLOAD_V1 = "execution_bound_catalog_tool_payload.v1"


class ExecutionBoundCatalogToolOperationPayload(SuspendedOperationPayload):
    """Exact catalog invocation intent for durable re-entry."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    payload_schema_version: str = Field(
        default=EXECUTION_BOUND_CATALOG_TOOL_PAYLOAD_V1,
        min_length=1,
    )
    tool_id: str = Field(min_length=1)
    tool_input_schema_id: str = Field(min_length=1)
    tool_input: CodeExecInput
    tenant_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    agent_id: str = Field(min_length=1)
    step_id: str = Field(min_length=1)
    invocation_scope_id: str = Field(min_length=1)
    idempotency_key: str = Field(min_length=1)
    correlation_request_id: str | None = None
    wiring_resolver_kind: str = Field(
        default="fixed_sandbox_session",
        min_length=1,
    )
    sandbox_session_id: str | None = None


__all__ = [
    "CODE_EXEC_INPUT_SCHEMA_ID",
    "EXECUTION_BOUND_CATALOG_TOOL_PAYLOAD_V1",
    "ExecutionBoundCatalogToolOperationPayload",
]
