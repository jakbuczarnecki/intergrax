# © Artur Czarnecki. All rights reserved.

"""Neutral agent runtime helpers for domain steps (scaffold + authoring)."""

from __future__ import annotations

from typing import Any

from intergrax.contracts.agent_step_context import AgentStepContext
from intergrax.contracts.runtime_execution_context import RuntimeExecutionContext
from intergrax.contracts.tool_request import ToolRequest, ToolResponseStatus
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
        metadata = attribute_access.optional(request, "metadata", None)
        return dict(metadata or {})
    if step_ctx is not None and fallback_keys:
        raw = step_ctx.metadata or {}
        return {key: raw[key] for key in fallback_keys if key in raw}
    return {}


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
