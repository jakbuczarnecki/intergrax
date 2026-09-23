# © Artur Czarnecki. All rights reserved.

"""Stable catalog tool invocation intent digest (excludes pause materialization metadata)."""

from __future__ import annotations

from intergrax.contracts.canonical_payload_hash import stable_payload_hash
from intergrax.contracts.execution_bound_catalog_tool_invocation import (
    ExecutionBoundCatalogToolInvokeRequest,
)
from intergrax.contracts.execution.suspended_operation.payload_catalog import (
    CODE_EXEC_INPUT_SCHEMA_ID,
    ExecutionBoundCatalogToolOperationPayload,
)
from intergrax.contracts.execution_identity import validate_run_id
from intergrax.tools.providers.sandbox.contracts import CodeExecInput
from intergrax.tools.invocation_wiring import durable_sandbox_session_id_from_resolver


def resolved_catalog_tool_idempotency_key(
    *,
    run_id: str,
    step_id: str,
    idempotency_key: str | None,
) -> str:
    if idempotency_key is not None and idempotency_key != "":
        return idempotency_key
    return f"{validate_run_id(run_id)}:{step_id}"


def digest_execution_bound_catalog_tool_invocation_intent(
    payload: ExecutionBoundCatalogToolOperationPayload,
) -> str:
    """Hash stable invocation intent only (not ``invocation_scope_id`` or pause artifacts)."""
    intent = {
        "tool_id": payload.tool_id,
        "tool_input_schema_id": payload.tool_input_schema_id,
        "tool_input": payload.tool_input.model_dump(mode="json"),
        "tenant_id": payload.tenant_id,
        "task_id": payload.task_id,
        "run_id": payload.run_id,
        "agent_id": payload.agent_id,
        "step_id": payload.step_id,
        "idempotency_key": payload.idempotency_key,
        "correlation_request_id": payload.correlation_request_id,
        "wiring_resolver_kind": payload.wiring_resolver_kind,
        "sandbox_session_id": payload.sandbox_session_id,
    }
    return stable_payload_hash(intent)


def digest_execution_bound_catalog_tool_invocation_intent_from_request(
    request: ExecutionBoundCatalogToolInvokeRequest,
) -> str:
    if type(request.input) is not CodeExecInput:
        raise TypeError("catalog invocation intent digest requires CodeExecInput")
    run_id_str = validate_run_id(request.run_id)
    sandbox_session_id = durable_sandbox_session_id_from_resolver(
        request.wiring_resolver,
    )
    idempotency_key = resolved_catalog_tool_idempotency_key(
        run_id=run_id_str,
        step_id=request.step_id,
        idempotency_key=request.idempotency_key,
    )
    intent_payload = {
        "tool_id": request.tool_id,
        "tool_input_schema_id": CODE_EXEC_INPUT_SCHEMA_ID,
        "tool_input": request.input.model_dump(mode="json"),
        "tenant_id": request.tenant_id,
        "task_id": str(request.task_id),
        "run_id": run_id_str,
        "agent_id": request.agent_id,
        "step_id": request.step_id,
        "idempotency_key": idempotency_key,
        "correlation_request_id": request.correlation_request_id,
        "wiring_resolver_kind": "fixed_sandbox_session",
        "sandbox_session_id": sandbox_session_id,
    }
    return stable_payload_hash(intent_payload)


__all__ = [
    "digest_execution_bound_catalog_tool_invocation_intent",
    "digest_execution_bound_catalog_tool_invocation_intent_from_request",
    "resolved_catalog_tool_idempotency_key",
]
