# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from pathlib import Path
from typing import Any

from intergrax.agents.authoring.runtime_tool_helpers import (
    allowlist_roots,
    exec_ctx_from_step,
    invoke_catalog_tool,
    parse_metadata_list,
    request_metadata,
    require_read_allowlist_roots,
    resolve_allowed_path,
    resolve_request_scope,
)
from intergrax.contracts.agent_step_context import AgentStepContext
from intergrax.contracts.runtime_execution_context import RAG_INGEST_TOOL_ID

INDEX_STEP_ID = "local_indexer_step"

_LKW_INDEX_METADATA_KEYS = frozenset(
    {
        "source_paths",
        "collection_id",
        "chunking_strategy_id",
        "tenant_id",
        "user_id",
        "workspace_id",
        "source_id",
        "document_id",
        "content_hash",
        "operation_id",
        "logical_source_path",
        "display_file_name",
    }
)

_INGEST_PROVENANCE_METADATA_KEYS = (
    "source_id",
    "document_id",
    "content_hash",
    "operation_id",
    "workspace_id",
)


def validate_allowlisted_files(
    paths: list[str],
    roots: frozenset[str],
) -> tuple[list[Path], list[dict[str, str]]]:
    allowed_roots = require_read_allowlist_roots(roots if roots else None)
    validated: list[Path] = []
    rejected: list[dict[str, str]] = []
    for raw in paths:
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
    metadata = request_metadata(exec_ctx, step_ctx, fallback_keys=_LKW_INDEX_METADATA_KEYS)
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
        workspace_id_raw = metadata.get("workspace_id")
        workspace_id = (
            str(workspace_id_raw).strip()
            if workspace_id_raw is not None and str(workspace_id_raw).strip()
            else collection_id
        )
        if collection_id:
            ingest_metadata["collection_id"] = collection_id
        if workspace_id:
            ingest_metadata["workspace_id"] = workspace_id
        if chunking := metadata.get("chunking_strategy_id"):
            ingest_metadata["chunking_strategy_id"] = chunking
        for key in _INGEST_PROVENANCE_METADATA_KEYS:
            if key == "workspace_id":
                continue
            raw = metadata.get(key)
            if raw is not None and str(raw).strip():
                ingest_metadata[key] = str(raw).strip()
        logical_source_path_raw = metadata.get("logical_source_path")
        logical_source_path = (
            str(logical_source_path_raw).strip()
            if logical_source_path_raw is not None and str(logical_source_path_raw).strip()
            else None
        )
        display_file_name_raw = metadata.get("display_file_name")
        display_file_name = (
            str(display_file_name_raw).strip()
            if display_file_name_raw is not None and str(display_file_name_raw).strip()
            else None
        )
        for path in validated:
            provenance_source_path = logical_source_path or str(path)
            provenance_file_name = display_file_name or path.name
            path_metadata = dict(ingest_metadata)
            path_metadata["source_path"] = provenance_source_path
            path_metadata["file_name"] = provenance_file_name
            tool_input: dict[str, Any] = {
                "source_path": str(path),
                "metadata": path_metadata,
            }
            if scope["tenant_id"]:
                tool_input["tenant_id"] = scope["tenant_id"]
            if scope["user_id"]:
                tool_input["user_id"] = scope["user_id"]
            if workspace_id:
                tool_input["workspace_id"] = workspace_id
            entry = await invoke_catalog_tool(
                exec_ctx,
                tool_name=RAG_INGEST_TOOL_ID,
                agent_id=step_ctx.agent_id,
                step_id=INDEX_STEP_ID,
                tool_input=tool_input,
            )
            entry["source_path"] = provenance_source_path
            entry["file_name"] = provenance_file_name
            ingested.append(entry)

    accepted_paths = [
        (
            str(metadata.get("logical_source_path")).strip()
            if metadata.get("logical_source_path") is not None
            and str(metadata.get("logical_source_path")).strip()
            else str(path)
        )
        for path in validated
    ]
    successful_ingests = [
        item
        for item in ingested
        if item.get("status") == "success" and item.get("used") is True
    ]
    num_chunks = sum(int(item.get("num_chunks") or 0) for item in successful_ingests)
    vector_ids: list[str] = []
    for item in successful_ingests:
        vector_ids.extend(list(item.get("vector_ids") or []))

    used = bool(successful_ingests)
    failed = [item for item in ingested if item not in successful_ingests]
    success_count = len(successful_ingests)
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
