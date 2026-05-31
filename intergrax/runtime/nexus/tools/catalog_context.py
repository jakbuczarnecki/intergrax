# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Invoke catalog context-injection tools from Nexus pipeline steps (Phase O.5b)."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel

from intergrax.llm.messages import ChatMessage
from intergrax.runtime.nexus.engine.runtime_state import RuntimeState
from intergrax.runtime.nexus.runtime_steps.tools import insert_context_before_last_user
from intergrax.tools.execution_models import ToolExecutionRequest
from intergrax.tools.unified.constants import RAG_RETRIEVE_TOOL_ID, WEBSEARCH_QUERY_TOOL_ID


def _invoker_registry(state: RuntimeState) -> Any | None:
    invoker = state.context.config.tool_invoker
    if invoker is None:
        return None
    registry = getattr(invoker, "registry", None)
    if registry is None:
        registry = getattr(invoker, "_base_invoker", None)
        if registry is not None:
            registry = getattr(registry, "registry", None) or getattr(registry, "_registry", None)
    return registry


def invoke_catalog_context_tool(
    state: RuntimeState,
    tool_id: str,
    input_payload: BaseModel,
    *,
    agent_id: str = "nexus",
    step_id: str = "catalog_context",
) -> bool:
    """
    Run a catalog tool when registered on the runtime invoker.

    Returns True when the catalog path was attempted (success or handled failure).
    Returns False when no invoker/registry entry — caller should use legacy step logic.
    """
    invoker = state.context.config.tool_invoker
    registry = _invoker_registry(state)
    if invoker is None or registry is None or not registry.has(tool_id):
        return False

    request = ToolExecutionRequest(
        run_id=state.run_id or "run",
        step_id=step_id,
        tool_id=tool_id,
        input=input_payload,
    )
    result = invoker.invoke(state=state, agent_id=agent_id, request=request)
    if not result.success or result.output is None:
        return True

    output = result.output
    used = bool(getattr(output, "used", True))
    context_text = str(getattr(output, "context_text", "") or "").strip()

    if tool_id == RAG_RETRIEVE_TOOL_ID:
        state.used_rag = used and bool(context_text)
    elif tool_id == WEBSEARCH_QUERY_TOOL_ID:
        state.used_websearch = used and bool(context_text)

    if used and context_text:
        _inject_context_text(state, tool_id=tool_id, context_text=context_text)

    return True


def _inject_context_text(state: RuntimeState, *, tool_id: str, context_text: str) -> None:
    label = "RAG CONTEXT" if tool_id == RAG_RETRIEVE_TOOL_ID else "WEB CONTEXT"
    state.tools_context_parts.append(f"{label}:\n{context_text}")
    insert_context_before_last_user(
        state,
        [ChatMessage(role="system", content=f"{label}:\n{context_text}")],
    )


def build_rag_retrieve_input(state: RuntimeState, *, top_k: Optional[int] = None) -> Any:
    from intergrax.tools.providers.rag.contracts import RagRetrieveInput

    cfg = state.context.config
    return RagRetrieveInput(
        query=(state.request.message or "").strip(),
        top_k=int(top_k or cfg.max_docs_per_query or 8),
        tenant_id=cfg.tenant_id,
        session_id=state.request.session_id,
        user_id=state.request.user_id,
        workspace_id=cfg.workspace_id,
    )


def build_websearch_query_input(state: RuntimeState, *, limit: Optional[int] = None) -> Any:
    from intergrax.tools.providers.websearch.contracts import WebsearchQueryInput

    cfg = state.context.config
    return WebsearchQueryInput(
        query=(state.request.message or "").strip(),
        limit=int(limit or cfg.max_docs_per_query or 8),
    )
