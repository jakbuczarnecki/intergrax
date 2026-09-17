# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Invoke catalog context-injection tools from ToolRuntime / on_next_step (Phase O.5b)."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Optional

from pydantic import BaseModel

from intergrax.runtime.nexus.engine.runtime_state import RuntimeState
from intergrax.runtime.nexus.context.context_builder import BuiltContext, RetrievedChunk
from intergrax.runtime.nexus.context.provider_handles import WEBSEARCH_BLOCKS_METADATA_KEY
from intergrax.runtime.nexus.context.runtime_state_handle_bridge import (
    merge_provider_metadata_into_request,
)
from intergrax.runtime.nexus.tools.context_injection_output import ContextInjectionOutput
from intergrax.runtime.nexus.tools.tool_invoker_protocol import ToolInvokerProtocol
from intergrax.tools.execution_models import ToolExecutionRequest
from intergrax.tools.providers.rag.contracts import RagRetrieveOutput
from intergrax.tools.providers.websearch.contracts import WebsearchQueryOutput
from intergrax.tools.unified.constants import RAG_RETRIEVE_TOOL_ID, WEBSEARCH_QUERY_TOOL_ID


def _resolve_invoker(state: RuntimeState) -> ToolInvokerProtocol | None:
    invoker = state.context.config.tool_invoker
    if invoker is None:
        return None
    if isinstance(invoker, ToolInvokerProtocol):
        return invoker
    return None


def _stage_rag_catalog_output(state: RuntimeState, output: RagRetrieveOutput) -> None:
    chunks: list[RetrievedChunk] = []
    for item in output.chunks:
        chunks.append(
            RetrievedChunk(
                id=item.id,
                text=item.text,
                metadata=dict(item.metadata or {}),
                score=float(item.score),
            )
        )
    if not chunks and output.context_text.strip():
        chunks.append(
            RetrievedChunk(
                id="catalog-rag-0",
                text=output.context_text.strip(),
                metadata={},
                score=0.0,
            )
        )
    if not chunks:
        return
    prior = state.context_builder_result
    if prior is None:
        state.context_builder_result = BuiltContext(
            history_messages=[],
            retrieved_chunks=chunks,
            rag_used=True,
            rag_reason=output.reason or "catalog",
        )
    else:
        state.context_builder_result = replace(
            prior,
            retrieved_chunks=chunks,
            rag_used=True,
            rag_reason=output.reason or prior.rag_reason or "catalog",
        )


def _stage_websearch_catalog_output(state: RuntimeState, output: WebsearchQueryOutput) -> None:
    blocks: list[dict[str, str]] = []
    for index, item in enumerate(output.results):
        body = (item.text or item.snippet or item.title or "").strip()
        if not body:
            continue
        blocks.append({"content": body, "source_id": f"web-{index}"})
    if not blocks and output.context_text.strip():
        blocks.append({"content": output.context_text.strip(), "source_id": "web-0"})
    if blocks:
        state.request.metadata[WEBSEARCH_BLOCKS_METADATA_KEY] = blocks


def _stage_catalog_context_for_ce(
    state: RuntimeState,
    *,
    tool_id: str,
    output: object,
    context_text: str,
) -> None:
    if tool_id == RAG_RETRIEVE_TOOL_ID and isinstance(output, RagRetrieveOutput):
        _stage_rag_catalog_output(state, output)
    elif tool_id == WEBSEARCH_QUERY_TOOL_ID and isinstance(output, WebsearchQueryOutput):
        _stage_websearch_catalog_output(state, output)
    elif context_text.strip():
        if tool_id == RAG_RETRIEVE_TOOL_ID:
            _stage_rag_catalog_output(
                state,
                RagRetrieveOutput(used=True, context_text=context_text.strip()),
            )
        elif tool_id == WEBSEARCH_QUERY_TOOL_ID:
            _stage_websearch_catalog_output(
                state,
                WebsearchQueryOutput(used=True, context_text=context_text.strip()),
            )
    merge_provider_metadata_into_request(state)


def invoke_catalog_context_tool(
    state: RuntimeState,
    tool_id: str,
    input_payload: BaseModel,
    *,
    step_id: str = "catalog_context",
) -> bool:
    """
    Run a catalog tool when registered on the runtime invoker.

    Returns True when the catalog path was attempted (success or handled failure).
    Returns False when no invoker/registry entry — caller should use legacy step logic.
    """
    invoker = _resolve_invoker(state)
    if invoker is None or not invoker.registry.has(tool_id):
        return False

    request = ToolExecutionRequest(
        run_id=state.run_id or "run",
        step_id=step_id,
        tool_id=tool_id,
        input=input_payload,
    )
    result = invoker.invoke(
        state=state,
        agent_id=state.request.agent_id,
        request=request,
    )
    if not result.success or result.output is None:
        return True

    output = result.output
    if not isinstance(output, ContextInjectionOutput):
        return True
    used = bool(output.used)
    context_text = str(output.context_text or "").strip()

    if tool_id == RAG_RETRIEVE_TOOL_ID:
        has_chunks = isinstance(output, RagRetrieveOutput) and bool(output.chunks)
        state.used_rag = used and (bool(context_text) or has_chunks)
    elif tool_id == WEBSEARCH_QUERY_TOOL_ID:
        has_results = isinstance(output, WebsearchQueryOutput) and bool(output.results)
        state.used_websearch = used and (bool(context_text) or has_results)

    if used and (context_text or isinstance(output, (RagRetrieveOutput, WebsearchQueryOutput))):
        _stage_catalog_context_for_ce(
            state,
            tool_id=tool_id,
            output=output,
            context_text=context_text,
        )

    return True


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
