# © Artur Czarnecki. All rights reserved.

"""Attach shadow/sandbox isolation to ``RuntimeExecutionContext`` and export refs."""

from __future__ import annotations

from typing import Any

from intergrax.contracts.runtime_execution_context import RuntimeExecutionContext
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.runtime.sandbox.manager import SandboxSessionManager
from intergrax.runtime.sandbox.sandbox_runtime import SANDBOX_SESSION_ID_KEY
from intergrax.runtime.task.task_metadata_bridge import execution_options_for_request
from intergrax.runtime.workspace.manager import ShadowWorkspaceManager
from intergrax.runtime.workspace.shadow_workspace import SHADOW_WORKSPACE_ID_KEY

RUNTIME_SHADOW_MANAGER_METADATA_KEY = "runtime.isolation.shadow_manager.v1"
RUNTIME_SANDBOX_MANAGER_METADATA_KEY = "runtime.isolation.sandbox_manager.v1"


def resolve_shadow_manager(metadata: dict[str, Any]) -> ShadowWorkspaceManager:
    manager = metadata.get(RUNTIME_SHADOW_MANAGER_METADATA_KEY)
    if isinstance(manager, ShadowWorkspaceManager):
        return manager
    return ShadowWorkspaceManager()


def resolve_sandbox_manager(metadata: dict[str, Any]) -> SandboxSessionManager:
    manager = metadata.get(RUNTIME_SANDBOX_MANAGER_METADATA_KEY)
    if isinstance(manager, SandboxSessionManager):
        return manager
    return SandboxSessionManager()


def attach_shadow_workspace_to_exec_ctx(
    exec_ctx: RuntimeExecutionContext,
    request: RuntimeRequest,
    *,
    shadow_manager: ShadowWorkspaceManager,
    task_id: str,
) -> None:
    if not execution_options_for_request(request).isolation.shadow_workspace:
        return
    tenant_id = request.tenant_id or "default"
    workspace = shadow_manager.open_or_create(tenant_id=tenant_id, task_id=task_id)
    exec_ctx.metadata["shadow_workspace"] = workspace
    exec_ctx.metadata[SHADOW_WORKSPACE_ID_KEY] = workspace.workspace_id


def attach_sandbox_session_to_exec_ctx(
    exec_ctx: RuntimeExecutionContext,
    request: RuntimeRequest,
    *,
    sandbox_manager: SandboxSessionManager,
    task_id: str,
) -> None:
    if not execution_options_for_request(request).isolation.sandbox:
        return
    tenant_id = request.tenant_id or "default"
    session = sandbox_manager.open_or_create(tenant_id=tenant_id, task_id=task_id)
    exec_ctx.metadata["sandbox_session"] = session
    exec_ctx.metadata[SANDBOX_SESSION_ID_KEY] = session.session_id


def attach_isolation_to_exec_ctx(
    exec_ctx: RuntimeExecutionContext,
    request: RuntimeRequest,
    *,
    task_id: str,
) -> None:
    """Open shadow/sandbox handles on ``exec_ctx`` when task isolation flags are set."""
    metadata = request.metadata or {}
    attach_shadow_workspace_to_exec_ctx(
        exec_ctx,
        request,
        shadow_manager=resolve_shadow_manager(metadata),
        task_id=task_id,
    )
    attach_sandbox_session_to_exec_ctx(
        exec_ctx,
        request,
        sandbox_manager=resolve_sandbox_manager(metadata),
        task_id=task_id,
    )


def isolation_structured_data_from_exec_ctx(
    exec_ctx: RuntimeExecutionContext | None,
) -> dict[str, Any]:
    """Export isolation handles for ``AgentExecutionResult.structured_data`` / bundle rollup."""
    if exec_ctx is None:
        return {}
    structured: dict[str, Any] = {}
    workspace_id = exec_ctx.metadata.get(SHADOW_WORKSPACE_ID_KEY)
    if workspace_id:
        structured[SHADOW_WORKSPACE_ID_KEY] = str(workspace_id)
        workspace = exec_ctx.metadata.get("shadow_workspace")
        if workspace is not None:
            structured["shadow_artifact_count"] = len(workspace.list_artifacts())
    session_id = exec_ctx.metadata.get(SANDBOX_SESSION_ID_KEY)
    if session_id:
        structured[SANDBOX_SESSION_ID_KEY] = str(session_id)
        session = exec_ctx.metadata.get("sandbox_session")
        if session is not None:
            structured["sandbox_operation_count"] = len(session.audit_log)
    return structured
