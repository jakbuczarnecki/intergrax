# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from pathlib import Path
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
from intergrax.tools.providers.rag.ingest_service import RAG_INGEST_TOOL_ID

INDEX_STEP_ID = "local_indexer_step"


def exec_ctx_from_step(step_ctx: AgentStepContext) -> RuntimeExecutionContext | None:
    raw = step_ctx.metadata.get("uaep_exec_ctx")
    if isinstance(raw, RuntimeExecutionContext):
        return raw
    return None


def request_metadata(exec_ctx: RuntimeExecutionContext | None) -> dict[str, Any]:
    if exec_ctx is None or exec_ctx.request is None:
        return {}
    request = exec_ctx.request
    if isinstance(request, RuntimeRequest):
        return dict(request.metadata or {})
    metadata = getattr(request, "metadata", None)
    return dict(metadata or {})


def allowlist_roots(exec_ctx: RuntimeExecutionContext | None) -> frozenset[str]:
    if exec_ctx is not None:
        runtime_state = exec_ctx.metadata.get("runtime_state")
        if runtime_state is not None:
            context = getattr(runtime_state, "context", None)
            config = getattr(context, "config", None) if context is not None else None
            wiring = getattr(config, "tool_wiring_context", None) if config is not None else None
            roots = getattr(wiring, "read_allowlist_roots", None) if wiring is not None else None
            if roots:
                return frozenset(roots)
    return read_allowlist_roots_from_env()


def parse_source_paths(metadata: dict[str, Any]) -> list[str]:
    raw = metadata.get("source_paths")
    if raw is None:
        return []
    if isinstance(raw, str):
        stripped = raw.strip()
        return [stripped] if stripped else []
    if isinstance(raw, (list, tuple)):
        paths: list[str] = []
        for item in raw:
            if isinstance(item, str) and item.strip():
                paths.append(item.strip())
        return paths
    return []


def validate_source_paths(
    source_paths: list[str],
    roots: frozenset[str],
) -> tuple[list[Path], list[dict[str, str]]]:
    allowed_roots = require_read_allowlist_roots(roots if roots else None)
    validated: list[Path] = []
    rejected: list[dict[str, str]] = []
    for raw in source_paths:
        try:
            resolved = resolve_allowed_path(raw, allowed_roots)
        except RuntimeError as exc:
            rejected.append({"path": raw, "reason": str(exc)})
            continue
        if not resolved.is_file():
            rejected.append({"path": raw, "reason": "source_not_found"})
            continue
        validated.append(resolved)
    return validated, rejected


async def ingest_document(
    exec_ctx: RuntimeExecutionContext,
    *,
    agent_id: str,
    step_id: str,
    source_path: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    ingest_metadata: dict[str, Any] = {}
    if collection_id := metadata.get("collection_id"):
        ingest_metadata["collection_id"] = collection_id
    if chunking := metadata.get("chunking_strategy_id"):
        ingest_metadata["chunking_strategy_id"] = chunking

    response = await exec_ctx.invoke_tool(
        ToolRequest(
            tool_name=RAG_INGEST_TOOL_ID,
            agent_id=agent_id,
            step_id=step_id,
            input={
                "source_path": source_path,
                "tenant_id": metadata.get("tenant_id"),
                "user_id": metadata.get("user_id"),
                "metadata": ingest_metadata,
            },
        )
    )
    entry: dict[str, Any] = {
        "source_path": source_path,
        "status": response.status.value,
    }
    if response.status == ToolResponseStatus.SUCCESS and response.output:
        entry.update(response.output)
    elif response.error:
        entry["reason"] = response.error
    return entry


def _failure_output(*, run_id: str, reason: str, rejected_paths: list[dict[str, str]] | None = None) -> dict[str, object]:
    answer = f"local_indexer: index failed — {reason}"
    return {
        "summary": answer,
        "answer": answer,
        "run_id": run_id,
        "ingest_summary": {
            "used": False,
            "reason": reason,
            "accepted_paths": [],
            "rejected_paths": rejected_paths or [],
            "ingested": [],
            "num_chunks": 0,
            "vector_ids": [],
        },
    }


async def run_index_job(step_ctx: AgentStepContext) -> dict[str, object]:
    exec_ctx = exec_ctx_from_step(step_ctx)
    metadata = request_metadata(exec_ctx)
    source_paths = parse_source_paths(metadata)

    if not source_paths:
        return _failure_output(run_id=step_ctx.run_id, reason="source_paths_missing")

    try:
        roots = allowlist_roots(exec_ctx)
        require_read_allowlist_roots(roots if roots else None)
    except RuntimeError as exc:
        return _failure_output(run_id=step_ctx.run_id, reason=str(exc))

    validated, rejected = validate_source_paths(source_paths, roots)
    ingested: list[dict[str, Any]] = []

    if exec_ctx is None:
        for path in validated:
            rejected.append({"path": str(path), "reason": "tool_gateway_not_available"})
    else:
        for path in validated:
            ingested.append(
                await ingest_document(
                    exec_ctx,
                    agent_id=step_ctx.agent_id,
                    step_id=INDEX_STEP_ID,
                    source_path=str(path),
                    metadata=metadata,
                )
            )

    accepted_paths = [str(path) for path in validated]
    num_chunks = sum(int(item.get("num_chunks") or 0) for item in ingested if item.get("status") == "success")
    vector_ids: list[str] = []
    for item in ingested:
        if item.get("status") == "success":
            vector_ids.extend(list(item.get("vector_ids") or []))

    used = any(item.get("status") == "success" for item in ingested)
    if ingested and not used:
        reason = "ingest_failed"
    elif rejected and not ingested:
        reason = "all_paths_rejected"
    elif used:
        reason = "ingest_complete"
    else:
        reason = "no_paths_ingested"

    summary_parts = [
        f"accepted={len(accepted_paths)}",
        f"rejected={len(rejected)}",
        f"ingested={sum(1 for item in ingested if item.get('status') == 'success')}",
        f"chunks={num_chunks}",
    ]
    answer = f"local_indexer: index job — {', '.join(summary_parts)}"
    return {
        "summary": answer,
        "answer": answer,
        "run_id": step_ctx.run_id,
        "ingest_summary": {
            "used": used,
            "reason": reason,
            "accepted_paths": accepted_paths,
            "rejected_paths": rejected,
            "ingested": ingested,
            "num_chunks": num_chunks,
            "vector_ids": vector_ids,
        },
    }
