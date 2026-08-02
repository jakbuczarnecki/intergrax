# © Artur Czarnecki. All rights reserved.

"""Sync RuntimeState artifacts into CE provider metadata keys (CE-HANDLE-FILL)."""

from __future__ import annotations

from typing import Any

from intergrax.runtime.nexus.context.context_handle_rows import ltm_entry_row, retrieved_chunk_row
from intergrax.runtime.nexus.context.provider_handles import (
    ATTACHMENT_SUMMARIES_METADATA_KEY,
    LTM_ENTRIES_METADATA_KEY,
    RAG_CHUNKS_METADATA_KEY,
    SESSION_HISTORY_MESSAGES_METADATA_KEY,
    SESSION_CONTEXT_REVISION_METADATA_KEY,
    SYSTEM_INSTRUCTIONS_METADATA_KEY,
    TOOL_OUTPUT_BLOCKS_METADATA_KEY,
    WEBSEARCH_BLOCKS_METADATA_KEY,
    try_build_session_history_snapshot_from_scope,
)
from intergrax.runtime.nexus.engine.runtime_state import RuntimeState


def extract_provider_metadata_from_runtime_state(state: RuntimeState) -> dict[str, Any]:
    """Build CE handle metadata dict from nexus ``RuntimeState`` artifacts."""
    metadata: dict[str, Any] = {}

    built = state.context_builder_result
    if built is not None and built.retrieved_chunks:
        metadata[RAG_CHUNKS_METADATA_KEY] = [
            retrieved_chunk_row(chunk) for chunk in built.retrieved_chunks
        ]

    ltm_result = state.user_longterm_memory_result
    if isinstance(ltm_result, dict):
        hits = ltm_result.get("hits")
        if isinstance(hits, list) and hits:
            metadata[LTM_ENTRIES_METADATA_KEY] = [ltm_entry_row(entry) for entry in hits]

    if state.base_history:
        snapshot = try_build_session_history_snapshot_from_scope(
            tenant_id=state.tenant_id,
            context_scope_id=state.request.session_id,
            revision_id=str(state.request.metadata.get(SESSION_CONTEXT_REVISION_METADATA_KEY) or "")
            or None,
            messages=list(state.base_history),
        )
        if snapshot is not None:
            from intergrax.context.session_history import SESSION_HISTORY_SNAPSHOT_HANDLE

            metadata[SESSION_HISTORY_SNAPSHOT_HANDLE] = snapshot
        else:
            metadata[SESSION_HISTORY_MESSAGES_METADATA_KEY] = list(state.base_history)
        revision_id = state.request.metadata.get(SESSION_CONTEXT_REVISION_METADATA_KEY)
        if revision_id:
            metadata[SESSION_CONTEXT_REVISION_METADATA_KEY] = revision_id

    vector_hits = state.request.metadata.get("session_vector_hits")
    if isinstance(vector_hits, list) and vector_hits:
        metadata["session_vector_hits"] = list(vector_hits)
    memory_profile = state.request.metadata.get("memory_profile")
    if isinstance(memory_profile, dict) and memory_profile.get("enable_session_vector_index"):
        metadata["memory_profile"] = dict(memory_profile)

    instruction_parts: list[str] = []
    if state.profile_org_instructions and state.profile_org_instructions.strip():
        instruction_parts.append(state.profile_org_instructions.strip())
    if state.profile_user_instructions and state.profile_user_instructions.strip():
        instruction_parts.append(state.profile_user_instructions.strip())
    if instruction_parts:
        metadata[SYSTEM_INSTRUCTIONS_METADATA_KEY] = "\n\n".join(instruction_parts)

    web_blocks: list[dict[str, Any]] = []
    tool_blocks: list[dict[str, Any]] = []
    for index, part in enumerate(state.tools_context_parts):
        text = str(part or "").strip()
        if not text:
            continue
        upper = text.upper()
        if upper.startswith("WEB CONTEXT"):
            body = text.split("\n", 1)[-1].strip() if "\n" in text else text
            web_blocks.append({"content": body, "source_id": f"web-{index}"})
        elif upper.startswith("RAG CONTEXT"):
            continue
        else:
            tool_blocks.append({"content": text, "source_id": f"tool-{index}"})
    if web_blocks:
        metadata[WEBSEARCH_BLOCKS_METADATA_KEY] = web_blocks
    if tool_blocks:
        metadata[TOOL_OUTPUT_BLOCKS_METADATA_KEY] = tool_blocks

    if state.ingestion_results:
        summaries = [
            {
                "attachment_id": result.attachment_id,
                "summary": (
                    f"Ingested {result.attachment_type}: "
                    f"{result.num_chunks} chunk(s)"
                ),
                "mime_type": result.attachment_type,
            }
            for result in state.ingestion_results
        ]
        if summaries:
            metadata[ATTACHMENT_SUMMARIES_METADATA_KEY] = summaries

    return metadata


def merge_provider_metadata_into_request(state: RuntimeState) -> dict[str, Any]:
    """Merge extracted provider metadata into ``state.request.metadata`` (in place)."""
    extracted = extract_provider_metadata_from_runtime_state(state)
    if extracted:
        state.request.metadata.update(extracted)
    return extracted
