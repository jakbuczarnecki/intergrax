# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Tier-3 MCP catalog Tool invoke → canonical ExecutionBoundCatalogToolInvoker (TLX-TL11 Gate B P07)."""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass
from typing import Any

from fastmcp.exceptions import ToolError
from pydantic import BaseModel, ValidationError

from intergrax.agents.tool_enablement import ToolEnablementProfile
from intergrax.applications._shared.mcp_harness_principal import resolve_mcp_run_agent_identity
from intergrax.applications._shared.mcp_host_execution_transport_cancel import (
    McpHostExecutionTransportCancelContext,
    McpProtectedHostExecutionAdmissionError,
    McpProtectedHostExecutionLaunch,
    create_mcp_protected_host_execution_launch,
    resolve_mcp_task_control_mutation_boundary_from_request,
    run_host_execution_with_mcp_transport_cancel,
)
from intergrax.contracts.agent_run import RequestIdentity
from intergrax.contracts.meaningful_side_effect_authorization import (
    MeaningfulSideEffectAuthorizationPort,
)
from intergrax.contracts.root_execution_transport_detachment_supervision import (
    RootExecutionTransportAdmissionRequest,
    RootExecutionTransportDetachmentHandoffDisposition,
    RootExecutionTransportDetachmentHandoffRefusalReason,
    RootExecutionTransportDetachmentSupervisionPort,
)
from intergrax.applications._shared.mcp_transport_detachment_supervision_coupling import (
    resolve_transport_detachment_supervision_from_request,
)
from intergrax.contracts.execution_bound_catalog_tool_invocation import (
    ExecutionBoundCatalogToolInvokeRequest,
    ExecutionBoundCatalogToolInvoker,
)
from intergrax.contracts.execution_identity import validate_run_id, validate_task_id
from intergrax.runtime.cancellation.coordinator import CancellationCoordinator
from intergrax.runtime.governance.active_governed_execution_task import (
    bind_governed_execution_task,
    reset_governed_execution_task,
)
from intergrax.runtime.governance.control_plane_mutation_authorization import (
    ControlPlaneMutationAuthorizationBoundary,
)
from intergrax.runtime.task.active_task_registry import ActiveTaskBinding, ActiveTaskRegistry
from intergrax.runtime.task.task import Task
from intergrax.tools.mcp_result_bounds import bound_mcp_structured_result
from intergrax.tools.registry.runtime import ToolRegistry

_METADATA_STEP_ID = "mcp_catalog_invoke_step_id"
_METADATA_AGENT_ID = "mcp_catalog_invoke_agent_id"
@dataclass(frozen=True, slots=True)
class McpCatalogInvocationComposition:
    """Sanctioned host composition inputs for MCP catalog Tool invoke (no new authority)."""

    catalog_invoker: ExecutionBoundCatalogToolInvoker
    registry: ToolRegistry
    tool_enablement: ToolEnablementProfile | None = None
    task_control_mutation_boundary: ControlPlaneMutationAuthorizationBoundary | None = None
    meaningful_side_effect_authorization: MeaningfulSideEffectAuthorizationPort | None = None
    default_tenant_id: str = "mcp"
    default_user_id: str = "mcp-user"


def _tool_error(code: str, detail: str) -> ToolError:
    return ToolError(f"{code}: {detail}")


def _bound_error_payload(code: str, detail: str, **extra: object) -> dict[str, object]:
    payload: dict[str, object] = {"error": code, "detail": detail}
    payload.update(extra)
    bounded, truncated = bound_mcp_structured_result(payload)
    if truncated or not isinstance(bounded, dict):
        return {"error": "mcp_payload_byte_limit_exceeded"}
    return bounded


async def resolve_mcp_catalog_execution_binding(
    *,
    task_id: str,
    asserted_run_id: str | None,
    resolved_tenant_id: str,
) -> ActiveTaskBinding:
    """Fail-closed admission: active registry binding is the only execution identity source."""
    try:
        validated_task_id = validate_task_id(task_id)
    except (TypeError, ValueError) as exc:
        raise _tool_error("invalid_task_id", str(exc)) from exc

    binding = await ActiveTaskRegistry.get(validated_task_id)
    if binding is None:
        raise _tool_error(
            "missing_execution_binding",
            f"no active execution binding for task_id={validated_task_id!r}",
        )

    if binding.task.tenant_id != resolved_tenant_id:
        raise _tool_error(
            "wrong_tenant",
            "caller tenant does not match active execution binding",
        )

    if asserted_run_id is not None:
        try:
            asserted = validate_run_id(asserted_run_id)
        except (TypeError, ValueError) as exc:
            raise _tool_error("invalid_run_id", str(exc)) from exc
        if str(asserted) != str(binding.run_id):
            raise _tool_error(
                "stale_run_binding",
                "asserted run_id does not match active execution binding",
            )

    return binding


def _assert_caller_authorized_for_execution(
    caller: RequestIdentity,
    task: Task,
) -> None:
    """Verified MCP principal must own the active execution (tenant match is insufficient)."""
    owner = (task.user_id or "").strip()
    caller_user = (caller.user_id or "").strip() or (caller.auth_subject or "").strip()
    if not owner:
        raise _tool_error(
            "missing_execution_binding",
            "active execution binding missing owner user_id",
        )
    if not caller_user:
        raise _tool_error(
            "missing_caller_identity",
            "verified caller missing user identity",
        )
    if owner != caller_user:
        raise _tool_error(
            "unauthorized_execution_caller",
            "verified caller is not the active execution owner",
        )


def _resolve_invoke_step_id(binding: Task, asserted_step_id: str | None) -> str:
    bound_step = binding.metadata.get(_METADATA_STEP_ID)
    if isinstance(bound_step, str) and bound_step.strip():
        canonical = bound_step.strip()
        if asserted_step_id is not None and asserted_step_id.strip() != canonical:
            raise _tool_error(
                "stale_step_binding",
                "asserted step_id does not match execution-bound catalog step",
            )
        return canonical
    raise _tool_error(
        "missing_execution_binding",
        "catalog invoke step_id not bound on active execution",
    )


def _resolve_invoke_agent_id(binding: Task) -> str:
    bound_agent = binding.metadata.get(_METADATA_AGENT_ID)
    if isinstance(bound_agent, str) and bound_agent.strip():
        return bound_agent.strip()
    if binding.agent_id and str(binding.agent_id).strip():
        return str(binding.agent_id).strip()
    raise _tool_error(
        "missing_execution_binding",
        "catalog invoke agent_id not bound on active execution",
    )


def _assert_catalog_tool_admitted(
    composition: McpCatalogInvocationComposition,
    tool_id: str,
) -> BaseModel:
    registry = composition.registry
    if not registry.has(tool_id):
        raise _tool_error("unknown_tool", tool_id)
    if composition.tool_enablement is not None and not composition.tool_enablement.is_tool_enabled(
        tool_id,
    ):
        raise _tool_error("tool_disabled", tool_id)
    return registry.get(tool_id).contract


def _assert_side_effect_invoke_admitted(
    composition: McpCatalogInvocationComposition,
    contract: object,
) -> None:
    if not getattr(contract, "side_effects", False):
        return
    if composition.meaningful_side_effect_authorization is None:
        raise _tool_error(
            "unauthorized_side_effect_tool",
            "side-effect catalog tool requires host meaningful side-effect authorization",
        )


def _parse_tool_input(contract: object, input_payload: dict[str, Any] | None) -> BaseModel:
    schema = getattr(contract, "input_schema", None)
    if schema is None or not isinstance(schema, type) or not issubclass(schema, BaseModel):
        raise _tool_error("invalid_tool_contract", "catalog tool input_schema missing")
    try:
        return schema.model_validate(input_payload or {})
    except ValidationError as exc:
        raise _tool_error("invalid_tool_input", str(exc)) from exc


def _serialize_tool_result(result: object) -> dict[str, object]:
    from intergrax.contracts.tool_execution_outcome import ToolExecutionResult

    if not isinstance(result, ToolExecutionResult):
        return _bound_error_payload("internal_error", "unexpected invoker result type")
    payload: dict[str, object] = {
        "tool_id": None,
        "success": result.success,
        "effect_certainty": result.effect_certainty.value,
    }
    if result.success:
        output = result.output
        if output is None:
            return _bound_error_payload("internal_error", "successful invoke missing output")
        payload["output"] = output.model_dump(mode="json")
    else:
        err = result.error
        payload["error_code"] = err.error_code if err is not None else "tool_failed"
        payload["error_message"] = err.error_message if err is not None else "tool_failed"
    bounded, truncated = bound_mcp_structured_result(payload)
    if truncated or not isinstance(bounded, dict):
        return {"error": "mcp_payload_byte_limit_exceeded"}
    return bounded


def _sync_invoke_catalog(
    composition: McpCatalogInvocationComposition,
    request: ExecutionBoundCatalogToolInvokeRequest,
    governed_task: Task,
) -> dict[str, object]:
    if CancellationCoordinator.is_requested(dict(governed_task.metadata)):
        return _bound_error_payload(
            "execution_cancelled",
            "active execution already has cooperative cancellation requested",
        )
    token = bind_governed_execution_task(governed_task)
    try:
        result = composition.catalog_invoker.invoke(request)
    finally:
        reset_governed_execution_task(token)
    if CancellationCoordinator.is_requested(dict(governed_task.metadata)):
        return _bound_error_payload(
            "execution_cancelled",
            "active execution cooperative cancellation observed after catalog invoke",
        )
    serialized = _serialize_tool_result(result)
    if isinstance(serialized.get("tool_id"), type(None)):
        serialized = {**serialized, "tool_id": request.tool_id}
    return serialized


async def _admit_catalog_invoke_host_work(
    *,
    supervision_port: RootExecutionTransportDetachmentSupervisionPort,
    binding: ActiveTaskBinding,
    caller: RequestIdentity,
    verified_tenant_id: str,
    launch: McpProtectedHostExecutionLaunch,
) -> None:
    """Open the catalog invoke start gate only after sanctioned supervision admission."""
    admission = await supervision_port.admit_root_execution_supervision(
        RootExecutionTransportAdmissionRequest(
            task_id=binding.task.task_id,
            verified_tenant_id=verified_tenant_id,
            caller_request_identity=caller,
            host_asyncio_task=launch.host_task,
        )
    )
    if admission.disposition is RootExecutionTransportDetachmentHandoffDisposition.ACCEPTED:
        launch.execution_start_gate.set()
        return
    if (
        admission.disposition is RootExecutionTransportDetachmentHandoffDisposition.REFUSED
        and admission.refusal_reason
        is RootExecutionTransportDetachmentHandoffRefusalReason.DUPLICATE_HANDOFF_CONFLICT
    ):
        launch.execution_start_gate.set()
        return
    launch.host_task.cancel()
    raise McpProtectedHostExecutionAdmissionError(
        "catalog invoke supervision admission refused before host execution",
    )


async def execute_mcp_catalog_tool_invocation(
    composition: McpCatalogInvocationComposition,
    *,
    tool_id: str,
    input_payload: dict[str, Any] | None = None,
    task_id: str,
    run_id: str | None = None,
    step_id: str | None = None,
    asserted_tenant_id: str | None = None,
) -> dict[str, object]:
    """
    Invoke one catalog Tool through the host-injected ExecutionBoundCatalogToolInvoker.

    Execution identity (tenant, run, step, agent) is taken from the active task binding
    and bound metadata — not synthesized from untrusted MCP fields alone.
    """
    identity = resolve_mcp_run_agent_identity(
        harness_auth=None,
        asserted_tenant_id=asserted_tenant_id,
        asserted_user_id=None,
        default_tenant_id=composition.default_tenant_id,
        default_user_id=composition.default_user_id,
    )
    if identity.request_identity is None:
        raise _tool_error(
            "missing_caller_identity",
            "authenticated harness principal required for catalog tool invoke",
        )

    assert identity.request_identity is not None
    binding = await resolve_mcp_catalog_execution_binding(
        task_id=task_id,
        asserted_run_id=run_id,
        resolved_tenant_id=identity.tenant_id,
    )
    _assert_caller_authorized_for_execution(identity.request_identity, binding.task)
    contract = _assert_catalog_tool_admitted(composition, tool_id)
    _assert_side_effect_invoke_admitted(composition, contract)
    parsed_input = _parse_tool_input(contract, input_payload)
    resolved_step = _resolve_invoke_step_id(binding.task, step_id)
    resolved_agent = _resolve_invoke_agent_id(binding.task)

    invoke_request = ExecutionBoundCatalogToolInvokeRequest(
        tool_id=tool_id,
        input=parsed_input,
        tenant_id=identity.tenant_id,
        task_id=str(binding.task_id),
        run_id=str(binding.run_id),
        agent_id=resolved_agent,
        step_id=resolved_step,
    )

    boundary = composition.task_control_mutation_boundary
    if boundary is None:
        boundary = resolve_mcp_task_control_mutation_boundary_from_request()
    cancel_ctx: McpHostExecutionTransportCancelContext | None = None
    if boundary is not None:
        cancel_ctx = McpHostExecutionTransportCancelContext(
            mutation_boundary=boundary,
            caller_request_identity=identity.request_identity,
        )

    async def _invoke_async() -> dict[str, object]:
        return await asyncio.to_thread(
            _sync_invoke_catalog,
            composition,
            invoke_request,
            binding.task,
        )

    launch = create_mcp_protected_host_execution_launch(_invoke_async)
    supervision = resolve_transport_detachment_supervision_from_request()

    if cancel_ctx is not None and supervision is not None:
        try:
            await _admit_catalog_invoke_host_work(
                supervision_port=supervision,
                binding=binding,
                caller=identity.request_identity,
                verified_tenant_id=identity.tenant_id,
                launch=launch,
            )
        except McpProtectedHostExecutionAdmissionError as exc:
            launch.host_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await launch.host_task
            raise _tool_error("execution_admission_denied", str(exc)) from exc
    else:
        launch.execution_start_gate.set()

    if cancel_ctx is None:
        try:
            return await launch.host_task
        except asyncio.CancelledError:
            launch.host_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await launch.host_task
            return _bound_error_payload(
                "transport_cancelled",
                "mcp catalog invoke waiter cancelled",
            )

    try:
        outcome = await run_host_execution_with_mcp_transport_cancel(
            launch.host_task,
            task=binding.task,
            cancel_ctx=cancel_ctx,
            transport_detachment_supervision=supervision,
            execution_start_gate=launch.execution_start_gate,
        )
    except McpProtectedHostExecutionAdmissionError as exc:
        raise _tool_error("execution_admission_denied", str(exc)) from exc
    except asyncio.CancelledError:
        return _bound_error_payload(
            "transport_cancelled",
            "mcp transport lost before catalog invoke completed",
        )
    if isinstance(outcome, dict):
        return outcome
    return _bound_error_payload("internal_error", "unexpected catalog invoke outcome")


__all__ = [
    "McpCatalogInvocationComposition",
    "execute_mcp_catalog_tool_invocation",
    "resolve_mcp_catalog_execution_binding",
]
