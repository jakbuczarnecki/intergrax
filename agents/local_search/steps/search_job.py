# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from typing import Any

from intergrax.contracts.acp_metadata_keys import AcpRunContextKey
from intergrax.contracts.agent_step_context import AgentStepContext
from intergrax.tools.providers.rag.service import RAG_TOOL_ID
from lkw_shared.runtime_helpers import (
    exec_ctx_from_step,
    invoke_catalog_tool,
    request_metadata,
    resolve_request_scope,
)

SEARCH_STEP_ID = "local_search_step"


def _optional_int(metadata: dict[str, Any], key: str) -> int | None:
    raw = metadata.get(key)
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _resolve_query(step_ctx: AgentStepContext, metadata: dict[str, Any]) -> str:
    run_input = step_ctx.metadata.get(AcpRunContextKey.RUN_INPUT, metadata.get("message", ""))
    if isinstance(run_input, dict):
        run_input = str(run_input.get("message") or run_input.get("summary") or "")
    return str(metadata.get("query") or run_input or step_ctx.message or "").strip()


def _format_evidence_item(
    *,
    text: str | None = None,
    content: str | None = None,
    source_path: str | None = None,
    chunk_id: str | None = None,
    score: float | None = None,
) -> dict[str, object]:
    item: dict[str, object] = {}
    body = (text or content or "").strip()
    if body:
        item["text"] = body
    if source_path:
        item["source_path"] = source_path
    if chunk_id:
        item["chunk_id"] = chunk_id
    if score is not None:
        item["score"] = score
    return item


def _format_evidence(chunks: list[Any], citations: list[Any]) -> list[dict[str, object]]:
    evidence: list[dict[str, object]] = []
    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        meta = chunk.get("metadata") if isinstance(chunk.get("metadata"), dict) else {}
        source_path = meta.get("source_path") or meta.get("source") or meta.get("file")
        item = _format_evidence_item(
            text=str(chunk.get("text") or ""),
            source_path=str(source_path) if source_path else None,
            chunk_id=str(chunk.get("id")) if chunk.get("id") is not None else None,
            score=chunk.get("score") if isinstance(chunk.get("score"), (int, float)) else None,
        )
        if item:
            evidence.append(item)

    if evidence:
        return evidence

    for citation in citations:
        if not isinstance(citation, dict):
            continue
        meta = citation.get("metadata") if isinstance(citation.get("metadata"), dict) else {}
        source_path = meta.get("source_path") or citation.get("source_label")
        item = _format_evidence_item(
            text=str(citation.get("excerpt") or ""),
            source_path=str(source_path) if source_path else None,
            chunk_id=str(citation.get("chunk_id")) if citation.get("chunk_id") is not None else None,
            score=citation.get("score") if isinstance(citation.get("score"), (int, float)) else None,
        )
        if item:
            evidence.append(item)
    return evidence


def _output(
    *,
    run_id: str,
    used: bool,
    reason: str,
    query: str = "",
    collection_id: str | None = None,
    evidence: list[dict[str, object]] | None = None,
    num_results: int = 0,
) -> dict[str, object]:
    num_results = num_results if used else 0
    evidence = evidence or []
    if used:
        answer = f"local_search: search job — query={query!r}, results={num_results}"
    else:
        answer = f"local_search: search failed — {reason}"
    return {
        "summary": answer,
        "answer": answer,
        "run_id": run_id,
        "search_summary": {
            "used": used,
            "reason": reason,
            "query": query,
            "collection_id": collection_id,
            "evidence": evidence,
            "num_results": num_results,
        },
    }


async def run_search_job(step_ctx: AgentStepContext) -> dict[str, object]:
    """LKW.1.2 — rag.retrieve via catalog tool; evidence-first search_summary."""
    exec_ctx = exec_ctx_from_step(step_ctx)
    metadata = request_metadata(exec_ctx)
    scope = resolve_request_scope(exec_ctx)
    query = _resolve_query(step_ctx, metadata)
    collection_id_raw = metadata.get("collection_id")
    collection_id = str(collection_id_raw).strip() if collection_id_raw is not None and str(collection_id_raw).strip() else None
    top_k = _optional_int(metadata, "top_k")

    if not query:
        return _output(run_id=step_ctx.run_id, used=False, reason="query_missing")

    if exec_ctx is None:
        return _output(
            run_id=step_ctx.run_id,
            used=False,
            reason="tool_gateway_not_available",
            query=query,
            collection_id=collection_id,
        )

    tool_input: dict[str, Any] = {"query": query}
    if top_k is not None:
        tool_input["top_k"] = top_k
    if scope["tenant_id"]:
        tool_input["tenant_id"] = scope["tenant_id"]
    if scope["user_id"]:
        tool_input["user_id"] = scope["user_id"]
    if collection_id:
        tool_input["workspace_id"] = collection_id

    entry = await invoke_catalog_tool(
        exec_ctx,
        tool_name=RAG_TOOL_ID,
        agent_id=step_ctx.agent_id,
        step_id=SEARCH_STEP_ID,
        tool_input=tool_input,
    )

    if entry.get("status") != "success" or not entry.get("used"):
        return _output(
            run_id=step_ctx.run_id,
            used=False,
            reason="retrieve_failed",
            query=query,
            collection_id=collection_id,
        )

    chunks = list(entry.get("chunks") or [])
    citations = list(entry.get("citations") or [])
    evidence = _format_evidence(chunks, citations)
    return _output(
        run_id=step_ctx.run_id,
        used=True,
        reason="retrieve_complete",
        query=query,
        collection_id=collection_id,
        evidence=evidence,
        num_results=len(evidence),
    )
