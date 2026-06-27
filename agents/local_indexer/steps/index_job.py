# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from typing import Any

from intergrax.contracts.agent_step_context import AgentStepContext
from lkw_shared.read_allowlist import require_read_allowlist_roots
from lkw_shared.tool_catalog import RAG_INGEST_TOOL_ID
from lkw_shared.runtime_helpers import (
    allowlist_roots,
    exec_ctx_from_step,
    invoke_catalog_tool,
    parse_metadata_list,
    request_metadata,
    resolve_request_scope,
    validate_allowlisted_files,
)

INDEX_STEP_ID = "local_indexer_step"


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
    metadata = request_metadata(exec_ctx, step_ctx)
    scope = resolve_request_scope(exec_ctx)
    source_paths = parse_metadata_list(metadata, "source_paths")

    if not source_paths:
        return _failure_output(run_id=step_ctx.run_id, reason="source_paths_missing")

    try:
        roots = allowlist_roots(exec_ctx)
        require_read_allowlist_roots(roots if roots else None)
    except RuntimeError as exc:
        return _failure_output(run_id=step_ctx.run_id, reason=str(exc))

    validated, rejected = validate_allowlisted_files(source_paths, roots)
    ingested: list[dict[str, Any]] = []

    if exec_ctx is None:
        for path in validated:
            rejected.append({"path": str(path), "reason": "tool_gateway_not_available"})
    else:
        ingest_metadata: dict[str, Any] = {}
        collection_id_raw = metadata.get("collection_id")
        collection_id = (
            str(collection_id_raw).strip()
            if collection_id_raw is not None and str(collection_id_raw).strip()
            else None
        )
        if collection_id:
            ingest_metadata["collection_id"] = collection_id
        if chunking := metadata.get("chunking_strategy_id"):
            ingest_metadata["chunking_strategy_id"] = chunking
        for path in validated:
            tool_input: dict[str, Any] = {
                "source_path": str(path),
                "metadata": ingest_metadata,
            }
            if scope["tenant_id"]:
                tool_input["tenant_id"] = scope["tenant_id"]
            if scope["user_id"]:
                tool_input["user_id"] = scope["user_id"]
            if collection_id:
                tool_input["workspace_id"] = collection_id
            entry = await invoke_catalog_tool(
                exec_ctx,
                tool_name=RAG_INGEST_TOOL_ID,
                agent_id=step_ctx.agent_id,
                step_id=INDEX_STEP_ID,
                tool_input=tool_input,
            )
            entry["source_path"] = str(path)
            ingested.append(entry)

    accepted_paths = [str(path) for path in validated]
    num_chunks = sum(int(item.get("num_chunks") or 0) for item in ingested if item.get("status") == "success")
    vector_ids: list[str] = []
    for item in ingested:
        if item.get("status") == "success":
            vector_ids.extend(list(item.get("vector_ids") or []))

    used = any(item.get("status") == "success" for item in ingested)
    failed = [item for item in ingested if item.get("status") != "success"]
    success_count = sum(1 for item in ingested if item.get("status") == "success")
    first_tool_reason = next(
        (str(item.get("reason")).strip() for item in failed if item.get("reason")),
        None,
    )
    if ingested and not used:
        reason = first_tool_reason or "ingest_failed"
    elif rejected and not ingested:
        reason = "all_paths_rejected"
    elif used:
        reason = "ingest_complete"
    else:
        reason = "no_paths_ingested"

    summary_parts = [
        f"accepted={len(accepted_paths)}",
        f"rejected={len(rejected)}",
        f"ingested={success_count}",
        f"chunks={num_chunks}",
    ]
    if failed:
        summary_parts.append(f"failed={len(failed)}")
        if first_tool_reason:
            summary_parts.append(f"tool_error={first_tool_reason}")
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
