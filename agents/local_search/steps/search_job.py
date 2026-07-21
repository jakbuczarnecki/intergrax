# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from typing import Any

from intergrax.agents.authoring.runtime_tool_helpers import (
    exec_ctx_from_step,
    invoke_catalog_tool,
    request_metadata,
    resolve_request_scope,
)
from intergrax.contracts.acp_metadata_keys import AcpRunContextKey
from intergrax.contracts.agent_step_context import AgentStepContext
from intergrax.contracts.runtime_execution_context import RAG_RETRIEVE_TOOL_ID
from local_search.diagnostics import SearchSummaryReason

SEARCH_STEP_ID = "local_search_step"

_LKW_SEARCH_METADATA_KEYS = frozenset(
    {
        "query",
        "collection_id",
        "top_k",
        "tenant_id",
        "user_id",
        "workspace_id",
    }
)


def _optional_int(metadata: dict[str, Any], key: str) -> int | None:
    raw = metadata.get(key)
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _resolve_query(step_ctx: AgentStepContext, metadata: dict[str, Any]) -> str:
    run_input = step_ctx.metadata.get(
        AcpRunContextKey.RUN_INPUT, metadata.get("message", "")
    )
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
    document_id: str | None = None,
    source_id: str | None = None,
    workspace_id: str | None = None,
    file_name: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, object]:
    item: dict[str, object] = {}
    body = (text or content or "").strip()
    if body:
        item["text"] = body
        item["snippet"] = body
    if source_path:
        item["source_path"] = source_path
    if chunk_id:
        item["chunk_id"] = chunk_id
    if score is not None:
        item["score"] = float(score)
    if document_id:
        item["document_id"] = document_id
    if source_id:
        item["source_id"] = source_id
    if workspace_id:
        item["workspace_id"] = workspace_id
    if file_name:
        item["file_name"] = file_name
    if metadata:
        item["metadata"] = metadata
    return item


def _format_evidence(
    chunks: list[Any],
    citations: list[Any],
    *,
    workspace_id: str | None = None,
) -> list[dict[str, object]]:
    evidence: list[dict[str, object]] = []
    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        metadata = chunk.get("metadata")
        meta: dict[str, Any] = metadata if isinstance(metadata, dict) else {}
        source_path = meta.get("source_path") or meta.get("source") or meta.get("file")
        file_name = meta.get("file_name")
        if not file_name and source_path:
            file_name = str(source_path).replace("\\", "/").rsplit("/", 1)[-1]
        raw_score = chunk.get("score")
        if not isinstance(raw_score, (int, float)):
            raw_score = meta.get("score")
        item_workspace = (
            str(meta["workspace_id"])
            if meta.get("workspace_id")
            else workspace_id
        )
        item = _format_evidence_item(
            text=str(chunk.get("text") or ""),
            source_path=str(source_path) if source_path else None,
            chunk_id=str(chunk.get("id")) if chunk.get("id") is not None else None,
            score=float(raw_score) if isinstance(raw_score, (int, float)) else None,
            document_id=str(meta["document_id"]) if meta.get("document_id") else None,
            source_id=str(meta["source_id"]) if meta.get("source_id") else None,
            workspace_id=item_workspace,
            file_name=str(file_name) if file_name else None,
            metadata=meta,
        )
        if item:
            evidence.append(item)

    if evidence:
        return evidence

    for citation in citations:
        if not isinstance(citation, dict):
            continue
        metadata = citation.get("metadata")
        meta = metadata if isinstance(metadata, dict) else {}
        source_path = meta.get("source_path") or citation.get("source_label")
        raw_score = citation.get("score")
        item = _format_evidence_item(
            text=str(citation.get("excerpt") or ""),
            source_path=str(source_path) if source_path else None,
            chunk_id=str(citation.get("chunk_id"))
            if citation.get("chunk_id") is not None
            else None,
            score=float(raw_score) if isinstance(raw_score, (int, float)) else None,
            document_id=str(meta["document_id"]) if meta.get("document_id") else None,
            source_id=str(meta["source_id"]) if meta.get("source_id") else None,
            workspace_id=str(meta["workspace_id"]) if meta.get("workspace_id") else workspace_id,
            file_name=str(meta["file_name"]) if meta.get("file_name") else None,
            metadata=meta,
        )
        if item:
            evidence.append(item)
    return evidence


def _output(
    *,
    run_id: str,
    used: bool,
    reason: SearchSummaryReason,
    query: str = "",
    collection_id: str | None = None,
    workspace_id: str | None = None,
    evidence: list[dict[str, object]] | None = None,
    num_results: int = 0,
) -> dict[str, object]:
    num_results = num_results if used else 0
    evidence = evidence or []
    if used:
        answer = f"local_search: search job — query={query!r}, results={num_results}"
    else:
        answer = f"local_search: search failed — {reason.value}"
    resolved_workspace = workspace_id or collection_id
    return {
        "summary": answer,
        "answer": answer,
        "run_id": run_id,
        "search_summary": {
            "used": used,
            "reason": reason.value,
            "query": query,
            "workspace_id": resolved_workspace,
            "collection_id": collection_id,
            "evidence": evidence,
            "num_results": num_results,
            "result_count": num_results,
        },
    }


def _resolved_tenant_id(
    scope: dict[str, str | None],
    metadata: dict[str, Any],
) -> str | None:
    tenant_id = scope.get("tenant_id")
    if tenant_id:
        return tenant_id
    raw = metadata.get("tenant_id")
    if raw is not None and str(raw).strip():
        return str(raw).strip()
    return None


async def run_search_job(step_ctx: AgentStepContext) -> dict[str, object]:
    """LKW.1.2 — rag.retrieve via catalog tool; evidence-first search_summary."""
    exec_ctx = exec_ctx_from_step(step_ctx)
    metadata = request_metadata(
        exec_ctx, step_ctx, fallback_keys=_LKW_SEARCH_METADATA_KEYS
    )
    scope = resolve_request_scope(exec_ctx)
    tenant_id = _resolved_tenant_id(scope, metadata)
    query = _resolve_query(step_ctx, metadata)
    collection_id_raw = metadata.get("collection_id")
    collection_id = (
        str(collection_id_raw).strip()
        if collection_id_raw is not None and str(collection_id_raw).strip()
        else None
    )
    workspace_id_raw = metadata.get("workspace_id")
    workspace_id = (
        str(workspace_id_raw).strip()
        if workspace_id_raw is not None and str(workspace_id_raw).strip()
        else collection_id
    )
    top_k = _optional_int(metadata, "top_k")

    if not query:
        return _output(
            run_id=step_ctx.run_id,
            used=False,
            reason=SearchSummaryReason.QUERY_MISSING,
        )

    if exec_ctx is None:
        return _output(
            run_id=step_ctx.run_id,
            used=False,
            reason=SearchSummaryReason.TOOL_GATEWAY_NOT_AVAILABLE,
            query=query,
            collection_id=collection_id,
        )

    tool_input: dict[str, Any] = {"query": query}
    if top_k is not None:
        tool_input["top_k"] = top_k
    if tenant_id:
        tool_input["tenant_id"] = tenant_id
    if scope["user_id"]:
        tool_input["user_id"] = scope["user_id"]
    if workspace_id:
        tool_input["workspace_id"] = workspace_id

    entry = await invoke_catalog_tool(
        exec_ctx,
        tool_name=RAG_RETRIEVE_TOOL_ID,
        agent_id=step_ctx.agent_id,
        step_id=SEARCH_STEP_ID,
        tool_input=tool_input,
    )

    if entry.get("status") != "success" or not entry.get("used"):
        summary = _output(
            run_id=step_ctx.run_id,
            used=False,
            reason=SearchSummaryReason.RETRIEVE_FAILED,
            query=query,
            collection_id=collection_id,
        )
        raw_reason = entry.get("reason")
        if raw_reason:
            raw_summary = summary["search_summary"]
            search_summary = dict(raw_summary) if isinstance(raw_summary, dict) else {}
            search_summary["raw_tool_reason"] = str(raw_reason)
            summary["search_summary"] = search_summary
        return summary

    chunks = list(entry.get("chunks") or [])
    citations = list(entry.get("citations") or [])
    evidence = _format_evidence(chunks, citations, workspace_id=workspace_id)
    return _output(
        run_id=step_ctx.run_id,
        used=True,
        reason=SearchSummaryReason.RETRIEVE_COMPLETE,
        query=query,
        collection_id=collection_id,
        workspace_id=workspace_id,
        evidence=evidence,
        num_results=len(evidence),
    )
