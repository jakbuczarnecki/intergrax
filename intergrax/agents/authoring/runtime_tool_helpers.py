# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Generic agent runtime helpers for catalog tool invocation and request scope."""

from __future__ import annotations

from typing import Any

from intergrax.contracts.agent_step_context import AgentStepContext
from intergrax.contracts.runtime_execution_context import RuntimeExecutionContext
from intergrax.contracts.tool_request import ToolRequest, ToolResponseStatus
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.tools.providers.filesystem.allowlist import (
    read_allowlist_roots_from_env,
    require_read_allowlist_roots,
    resolve_allowed_path,
)
from intergrax.utils import attribute_access


def exec_ctx_from_step(step_ctx: AgentStepContext) -> RuntimeExecutionContext | None:
    raw = step_ctx.metadata.get("uaep_exec_ctx")
    if isinstance(raw, RuntimeExecutionContext):
        return raw
    return None


def request_metadata(
    exec_ctx: RuntimeExecutionContext | None,
    step_ctx: AgentStepContext | None = None,
    *,
    fallback_keys: frozenset[str] | None = None,
) -> dict[str, Any]:
    if exec_ctx is not None and exec_ctx.request is not None:
        request = exec_ctx.request
        if isinstance(request, RuntimeRequest):
            return dict(request.metadata or {})
        metadata = attribute_access.optional(request, "metadata", None)
        return dict(metadata or {})
    if step_ctx is not None and fallback_keys:
        raw = step_ctx.metadata or {}
        return {key: raw[key] for key in fallback_keys if key in raw}
    return {}


def resolve_request_scope(exec_ctx: RuntimeExecutionContext | None) -> dict[str, str | None]:
    """Authoritative tenant/user scope for catalog tool calls.

    ``RuntimeRequest.tenant_id`` / ``user_id`` win over ``request.metadata`` fields.
    Conflicting ``metadata.tenant_id`` values are ignored (not propagated to tools).
    """
    if exec_ctx is None or exec_ctx.request is None:
        return {"tenant_id": None, "user_id": None}

    request = exec_ctx.request
    metadata = request_metadata(exec_ctx)
    meta_tenant = metadata.get("tenant_id")
    meta_user = metadata.get("user_id")

    if isinstance(request, RuntimeRequest):
        tenant_id = request.tenant_id if request.tenant_id and str(request.tenant_id).strip() else None
        user_id = request.user_id if request.user_id and str(request.user_id).strip() else None
    else:
        tenant_id = attribute_access.optional(request, "tenant_id", None)
        user_id = attribute_access.optional(request, "user_id", None)
        tenant_id = str(tenant_id).strip() if tenant_id and str(tenant_id).strip() else None
        user_id = str(user_id).strip() if user_id and str(user_id).strip() else None

    if tenant_id is None and meta_tenant is not None and str(meta_tenant).strip():
        tenant_id = str(meta_tenant).strip()
    if user_id is None and meta_user is not None and str(meta_user).strip():
        user_id = str(meta_user).strip()

    return {"tenant_id": tenant_id, "user_id": user_id}


def allowlist_roots(exec_ctx: RuntimeExecutionContext | None) -> frozenset[str]:
    if exec_ctx is not None:
        runtime_state = exec_ctx.metadata.get("runtime_state")
        if runtime_state is not None:
            context = attribute_access.optional(runtime_state, "context", None)
            config = attribute_access.optional(context, "config", None) if context is not None else None
            wiring = attribute_access.optional(config, "tool_wiring_context", None) if config is not None else None
            roots = attribute_access.optional(wiring, "read_allowlist_roots", None) if wiring is not None else None
            if roots:
                return frozenset(roots)
    return read_allowlist_roots_from_env()


def parse_metadata_list(metadata: dict[str, Any], key: str) -> list[str]:
    raw = metadata.get(key)
    if raw is None:
        return []
    if isinstance(raw, str):
        stripped = raw.strip()
        return [stripped] if stripped else []
    if isinstance(raw, (list, tuple)):
        values: list[str] = []
        for item in raw:
            if isinstance(item, str) and item.strip():
                values.append(item.strip())
        return values
    return []


async def invoke_catalog_tool(
    exec_ctx: RuntimeExecutionContext,
    *,
    tool_name: str,
    agent_id: str,
    step_id: str,
    tool_input: dict[str, Any],
) -> dict[str, Any]:
    response = await exec_ctx.invoke_tool(
        ToolRequest(
            tool_name=tool_name,
            agent_id=agent_id,
            step_id=step_id,
            input=tool_input,
        )
    )
    entry: dict[str, Any] = {"status": response.status.value}
    if response.status == ToolResponseStatus.SUCCESS and response.output:
        entry.update(response.output)
    elif response.error:
        entry["reason"] = response.error
    elif response.status != ToolResponseStatus.SUCCESS:
        entry["reason"] = response.status.value
    return entry
