# © Artur Czarnecki. All rights reserved.

"""Sync RuntimeState artifacts into CE provider metadata keys (CE-HANDLE-FILL)."""

from __future__ import annotations

from typing import Any

from intergrax.runtime.nexus.context.provider_handles import (
    ATTACHMENT_SUMMARIES_METADATA_KEY,
    LTM_ENTRIES_METADATA_KEY,
    RAG_CHUNKS_METADATA_KEY,
    SESSION_HISTORY_MESSAGES_METADATA_KEY,
    SYSTEM_INSTRUCTIONS_METADATA_KEY,
    TOOL_OUTPUT_BLOCKS_METADATA_KEY,
    WEBSEARCH_BLOCKS_METADATA_KEY,
)
from intergrax.runtime.nexus.engine.runtime_state import RuntimeState


def _retrieved_chunk_row(chunk: Any) -> dict[str, Any]:
    if isinstance(chunk, dict):
        metadata = dict(chunk.get("metadata") or {})
        if chunk.get("id") is not None:
            metadata.setdefault("id", chunk.get("id"))
        if chunk.get("score") is not None:
            metadata.setdefault("score", chunk.get("score"))
        text = str(chunk.get("text") or chunk.get("content") or "").strip()
        return {"text": text, "metadata": metadata}
    metadata = dict(getattr(chunk, "metadata", None) or {})
    chunk_id = getattr(chunk, "id", None)
    if chunk_id is not None:
        metadata.setdefault("id", chunk_id)
    score = getattr(chunk, "score", None)
    if score is not None:
        metadata.setdefault("score", score)
    text = str(getattr(chunk, "text", "") or "").strip()
    return {"text": text, "metadata": metadata}


def _ltm_entry_row(entry: Any) -> dict[str, Any]:
    if isinstance(entry, dict):
        return dict(entry)
    row: dict[str, Any] = {}
    for key in ("entry_id", "content", "title", "session_id", "kind", "importance", "deleted"):
        if hasattr(entry, key):
            row[key] = getattr(entry, key)
    if not row.get("content") and hasattr(entry, "content"):
        row["content"] = str(getattr(entry, "content") or "")
    return row


def extract_provider_metadata_from_runtime_state(state: RuntimeState) -> dict[str, Any]:
    """Build CE handle metadata dict from nexus ``RuntimeState`` artifacts."""
    metadata: dict[str, Any] = {}

    built = state.context_builder_result
    if built is not None and built.retrieved_chunks:
        metadata[RAG_CHUNKS_METADATA_KEY] = [
            _retrieved_chunk_row(chunk) for chunk in built.retrieved_chunks
        ]

    ltm_result = state.user_longterm_memory_result
    if ltm_result is not None:
        entries = getattr(ltm_result, "retrieved_entries", None)
        if isinstance(entries, list) and entries:
            metadata[LTM_ENTRIES_METADATA_KEY] = [_ltm_entry_row(entry) for entry in entries]

    if state.base_history:
        metadata[SESSION_HISTORY_MESSAGES_METADATA_KEY] = list(state.base_history)

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
