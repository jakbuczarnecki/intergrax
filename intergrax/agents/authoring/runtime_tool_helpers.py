# © Artur Czarnecki. All rights reserved.

"""Neutral agent runtime helpers for domain steps (scaffold + authoring)."""

from __future__ import annotations

from typing import Any

from intergrax.contracts.agent_step_context import AgentStepContext
from intergrax.contracts.runtime_execution_context import RuntimeExecutionContext
from intergrax.contracts.tool_request import ToolRequest, ToolResponseStatus
from intergrax.runtime.execution.agent_runtime_io import (
    RuntimeRequest,
    canonical_runtime_request_tenant_id,
)
from intergrax.tools.providers.filesystem.allowlist import (
    read_allowlist_roots_from_env,
    require_read_allowlist_roots,
    resolve_allowed_path,
)
from intergrax.utils import attribute_access


class RequestScopeError(ValueError):
    """Catalog tool request scope could not be resolved (fail-closed)."""


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
        if isinstance(metadata, dict):
            return dict(metadata)
        return {}
    if step_ctx is not None and fallback_keys:
        raw = step_ctx.metadata or {}
        return {key: raw[key] for key in fallback_keys if key in raw}
    return {}


def resolve_request_scope(exec_ctx: RuntimeExecutionContext | None) -> dict[str, str | None]:
    """Project typed execution tenant/user into catalog tool scope (never from metadata authority)."""
    if exec_ctx is None or exec_ctx.request is None:
        return {"tenant_id": None, "user_id": None}

    request = exec_ctx.request
    metadata = request_metadata(exec_ctx)
    meta_tenant = metadata.get("tenant_id")
    meta_user = metadata.get("user_id")

    if isinstance(request, RuntimeRequest):
        try:
            tenant_id: str | None = canonical_runtime_request_tenant_id(request)
        except ValueError as exc:
            raise RequestScopeError(str(exc)) from exc
        user_id = request.user_id if request.user_id and str(request.user_id).strip() else None
    else:
        tenant_id = None
        user_id = attribute_access.optional(request, "user_id", None)
        user_id = str(user_id).strip() if user_id and str(user_id).strip() else None
        if meta_tenant is not None and str(meta_tenant).strip():
            raise RequestScopeError(
                "metadata tenant_id cannot establish catalog tenant authority"
            )

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
            if isinstance(roots, (list, tuple, set, frozenset)) and roots:
                return frozenset(str(root) for root in roots)
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


__all__ = [
    "RequestScopeError",
    "allowlist_roots",
    "exec_ctx_from_step",
    "invoke_catalog_tool",
    "parse_metadata_list",
    "request_metadata",
    "require_read_allowlist_roots",
    "resolve_allowed_path",
    "resolve_request_scope",
]
