# © Artur Czarnecki. All rights reserved.

"""Legacy collector adapters for builtin context providers (CE-PROV-BRIDGE).

Handle key contract (``ContextProviderContext.handles``):

| Key | Type | Used by |
|-----|------|---------|
| ``session_history_snapshot`` | ``SessionHistorySnapshot`` | ``builtin.session_history`` (canonical) |
| ``prior_output_records`` | ``list[PriorOutputRecord]`` or dict rows | ``builtin.graph_prior`` |
| ``session_history_messages`` | ``list[ChatMessage]`` | legacy compatibility only |
| ``messages`` | ``list[ChatMessage]`` | fallback for ``builtin.task_message`` |
| ``rag_chunks`` | list of chunk rows | ``builtin.rag`` |
| ``ltm_entries`` | list of LTM entry rows | ``builtin.longterm_memory`` |
| ``websearch_blocks`` | list[str] or dict blocks | ``builtin.websearch`` |
| ``tool_output_blocks`` | list[str] or dict blocks | ``builtin.tool_output`` |
| ``system_instructions`` | str | ``builtin.system_instructions`` |
| ``policy_overlay_fragments`` | list[dict] with overlay_id/content | ``builtin.policy_overlay`` |
| ``attachment_summaries`` | list[dict] with attachment_id/summary | ``builtin.attachments`` |
| ``shared_context_reads`` | dict keyed by shared context entry | ``builtin.shared_context`` |

``ContextAssemblyRequest.objective`` is the primary task-message source on graph/UAEP paths.
"""

from __future__ import annotations
from intergrax.utils import attribute_access

from typing import Any

from intergrax.context.contracts import (
    ContextAssemblyRequest,
    ContextFragment,
    ContextFragmentSource,
    IterativeToolOutputBlock,
    content_hash_for_text,
)
from intergrax.context.session_history import (
    SessionHistorySnapshotRequiredError,
    require_session_history_messages,
)
from intergrax.llm.messages import ChatMessage

PRIOR_OUTPUT_RECORDS_HANDLE = "prior_output_records"
SESSION_HISTORY_MESSAGES_HANDLE = "session_history_messages"
RAG_CHUNKS_HANDLE = "rag_chunks"
LTM_ENTRIES_HANDLE = "ltm_entries"
WEBSEARCH_BLOCKS_HANDLE = "websearch_blocks"
TOOL_OUTPUT_BLOCKS_HANDLE = "tool_output_blocks"
SYSTEM_INSTRUCTIONS_HANDLE = "system_instructions"
POLICY_OVERLAY_FRAGMENTS_HANDLE = "policy_overlay_fragments"
ATTACHMENT_SUMMARIES_HANDLE = "attachment_summaries"
SHARED_CONTEXT_READS_HANDLE = "shared_context_reads"

_DEFAULT_SCORES = {
    ContextFragmentSource.TASK_MESSAGE: (0.95, 0.9, 0.95),
    ContextFragmentSource.GRAPH_PRIOR: (0.85, 0.8, 0.85),
    ContextFragmentSource.SESSION_HISTORY: (0.75, 0.7, 0.8),
    ContextFragmentSource.SYSTEM_INSTRUCTIONS: (1.0, 1.0, 1.0),
    ContextFragmentSource.LONGTERM_MEMORY: (0.8, 0.75, 0.85),
    ContextFragmentSource.RAG: (0.85, 0.8, 0.9),
    ContextFragmentSource.WEBSEARCH: (0.75, 0.9, 0.75),
    ContextFragmentSource.TOOL_OUTPUT: (0.9, 0.95, 0.9),
    ContextFragmentSource.SHARED_CONTEXT: (0.8, 0.85, 0.85),
    ContextFragmentSource.ATTACHMENT: (0.7, 0.9, 0.8),
    ContextFragmentSource.POLICY_OVERLAY: (0.95, 1.0, 0.95),
}


def _scores_for(source: ContextFragmentSource) -> tuple[float, float, float]:
    return _DEFAULT_SCORES.get(source, (0.7, 0.7, 0.7))


def _fragment(
    *,
    fragment_id: str,
    source: ContextFragmentSource,
    source_id: str,
    content: str,
    mandatory: bool = False,
    metadata: dict[str, Any] | None = None,
) -> ContextFragment:
    relevance, freshness, confidence = _scores_for(source)
    return ContextFragment(
        fragment_id=fragment_id,
        source=source,
        source_id=source_id,
        content=content,
        token_estimate=max(1, len(content) // 4),
        relevance_score=relevance,
        freshness_score=freshness,
        confidence_score=confidence,
        mandatory=mandatory,
        metadata=dict(metadata or {}),
        content_hash=content_hash_for_text(content),
    )


def fragments_from_task_message(
    request: ContextAssemblyRequest,
    *,
    messages: list[ChatMessage] | None = None,
) -> list[ContextFragment]:
    """Emit TASK_MESSAGE fragment from request objective or last user turn."""
    if ContextFragmentSource.TASK_MESSAGE in request.excluded_sources:
        return []
    text = (request.objective or "").strip()
    if not text and messages:
        for message in reversed(messages):
            if message.role == "user" and (message.content or "").strip():
                text = (message.content or "").strip()
                break
    if not text:
        return []
    return [
        _fragment(
            fragment_id=f"task-{request.task_id}",
            source=ContextFragmentSource.TASK_MESSAGE,
            source_id=request.task_id,
            content=text,
            mandatory=True,
            metadata={"assembly_scope": request.assembly_scope},
        )
    ]


def _record_text(record: Any) -> str:
    if hasattr(record, "evidence") and (attribute_access.optional(record, "evidence") or "").strip():
        return str(record.evidence).strip()
    if hasattr(record, "summary"):
        return str(record.summary or "").strip()
    if isinstance(record, dict):
        evidence = str(record.get("evidence") or "").strip()
        if evidence:
            return evidence
        return str(record.get("summary") or "").strip()
    return ""


def _record_node_id(record: Any) -> str:
    if hasattr(record, "node_id"):
        return str(record.node_id)
    if isinstance(record, dict):
        return str(record.get("node_id") or "unknown")
    return "unknown"


def _record_agent_id(record: Any) -> str:
    if hasattr(record, "agent_id"):
        return str(record.agent_id or "")
    if isinstance(record, dict):
        return str(record.get("agent_id") or "")
    return ""


def fragments_from_prior_output_records(
    records: list[Any],
    *,
    max_entries: int | None = None,
) -> list[ContextFragment]:
    """Emit GRAPH_PRIOR fragments from ``collect_dependency_records`` rows."""
    if not records:
        return []
    limit = max_entries if max_entries is not None else len(records)
    fragments: list[ContextFragment] = []
    for index, record in enumerate(records[:limit]):
        text = _record_text(record)
        if not text:
            continue
        node_id = _record_node_id(record)
        agent_id = _record_agent_id(record)
        fragments.append(
            _fragment(
                fragment_id=f"graph-prior-{node_id}-{index}",
                source=ContextFragmentSource.GRAPH_PRIOR,
                source_id=node_id,
                content=text,
                metadata={"agent_id": agent_id, "node_id": node_id},
            )
        )
    return fragments


def fragments_from_session_history(
    messages: list[Any],
    *,
    max_entries: int = 8,
    include_session_history: bool = True,
) -> list[ContextFragment]:
    """Legacy compatibility shim.

    Non-empty history must use SessionHistorySnapshot.
    """
    _ = max_entries

    if not include_session_history:
        return []
    if messages is None or messages == []:
        return []

    validated = require_session_history_messages(messages)
    if not validated:
        return []
    raise SessionHistorySnapshotRequiredError()


def _chunk_text(ch: Any) -> str:
    if isinstance(ch, str):
        return ch.strip()
    if isinstance(ch, dict):
        for key in ("text", "content", "page_content", "chunk", "value"):
            value = ch.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return ""
    for attr in ("text", "content", "page_content", "chunk", "value"):
        if hasattr(ch, attr):
            value = attribute_access.optional(ch, attr)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return ""


def _chunk_metadata(ch: Any) -> dict[str, Any]:
    if isinstance(ch, dict):
        meta = ch.get("metadata")
        if isinstance(meta, dict):
            return dict(meta)
        return {key: value for key, value in ch.items() if key not in {"text", "content"}}
    if hasattr(ch, "metadata") and isinstance(ch.metadata, dict):
        return dict(ch.metadata)
    return {}


def _block_text(block: Any) -> str:
    if isinstance(block, str):
        return block.strip()
    if isinstance(block, dict):
        for key in ("content", "text", "summary", "body"):
            value = block.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return _chunk_text(block)


def _block_source_id(block: Any, *, fallback: str) -> str:
    if isinstance(block, dict):
        for key in ("source_id", "id", "tool_call_id", "overlay_id", "attachment_id"):
            value = block.get(key)
            if value is not None and str(value).strip():
                return str(value).strip()
    for attr in ("source_id", "id", "tool_call_id", "overlay_id", "attachment_id"):
        if hasattr(block, attr):
            value = attribute_access.optional(block, attr)
            if value is not None and str(value).strip():
                return str(value).strip()
    return fallback


def fragments_from_rag_chunks(
    chunks: list[Any],
    *,
    max_chars: int = 4000,
) -> list[ContextFragment]:
    """Emit RAG fragments — one per chunk with citation metadata."""
    if not chunks:
        return []
    fragments: list[ContextFragment] = []
    total_chars = 0
    for index, chunk in enumerate(chunks):
        text = _chunk_text(chunk)
        if not text:
            continue
        meta = _chunk_metadata(chunk)
        chunk_id = str(meta.get("id") or meta.get("doc_id") or meta.get("chunk_id") or index)
        if total_chars + len(text) > max_chars:
            remaining = max_chars - total_chars
            if remaining <= 80:
                break
            text = text[:remaining].rstrip() + "…"
        total_chars += len(text)
        citations = []
        for key in ("source", "url", "doc_id", "file", "page", "page_number"):
            if key in meta and meta[key] is not None:
                citations.append({key: meta[key]})
        fragments.append(
            _fragment(
                fragment_id=f"rag-{chunk_id}-{index}",
                source=ContextFragmentSource.RAG,
                source_id=str(chunk_id),
                content=text,
                metadata={"citations": citations, **meta},
            )
        )
        if total_chars >= max_chars:
            break
    return fragments


def fragments_from_ltm_entries(
    entries: list[Any],
    *,
    max_entries: int = 12,
) -> list[ContextFragment]:
    """Emit LONGTERM_MEMORY fragments from retrieved profile memory rows."""
    if not entries:
        return []
    fragments: list[ContextFragment] = []
    for index, entry in enumerate(entries[:max_entries]):
        if isinstance(entry, dict) and entry.get("deleted"):
            continue
        if hasattr(entry, "deleted") and attribute_access.optional(entry, "deleted"):
            continue
        text = _chunk_text(entry)
        if not text:
            continue
        entry_id = _block_source_id(entry, fallback=f"ltm-{index}")
        meta: dict[str, Any] = {}
        if isinstance(entry, dict):
            meta = {
                key: entry[key]
                for key in ("kind", "title", "session_id", "importance")
                if key in entry
            }
        else:
            for key in ("kind", "title", "session_id", "importance"):
                if hasattr(entry, key):
                    meta[key] = attribute_access.optional(entry, key)
        fragments.append(
            _fragment(
                fragment_id=f"ltm-{entry_id}",
                source=ContextFragmentSource.LONGTERM_MEMORY,
                source_id=str(entry_id),
                content=text,
                metadata=meta,
            )
        )
    return fragments


def fragments_from_websearch_blocks(
    blocks: list[Any],
    *,
    max_blocks: int = 8,
) -> list[ContextFragment]:
    """Emit WEBSEARCH fragments from serialized search result blocks."""
    if not blocks:
        return []
    fragments: list[ContextFragment] = []
    for index, block in enumerate(blocks[:max_blocks]):
        text = _block_text(block)
        if not text:
            continue
        source_id = _block_source_id(block, fallback=f"web-{index}")
        meta: dict[str, Any] = {}
        if isinstance(block, dict):
            meta = {
                key: block[key]
                for key in ("url", "title", "snippet")
                if key in block
            }
        fragments.append(
            _fragment(
                fragment_id=f"websearch-{source_id}",
                source=ContextFragmentSource.WEBSEARCH,
                source_id=str(source_id),
                content=text,
                metadata=meta,
            )
        )
    return fragments


def fragment_from_iterative_tool_output_block(
    block: IterativeToolOutputBlock,
) -> ContextFragment:
    """Canonical TOOL_OUTPUT fragment from a typed iterative tool-feedback block."""
    metadata: dict[str, Any] = {
        "tool_call_id": block.tool_call_id,
        "tool_name": block.tool_name,
    }
    if block.step_id is not None:
        metadata["step_id"] = block.step_id
    return _fragment(
        fragment_id=f"tool-output-{block.tool_call_id}",
        source=ContextFragmentSource.TOOL_OUTPUT,
        source_id=block.tool_call_id,
        content=block.content,
        metadata=metadata,
    )


def fragments_from_tool_output_blocks(
    blocks: list[Any],
    *,
    max_blocks: int = 16,
) -> list[ContextFragment]:
    """Emit TOOL_OUTPUT fragments from step tool result blocks."""
    if not blocks:
        return []
    fragments: list[ContextFragment] = []
    for index, block in enumerate(blocks[:max_blocks]):
        if isinstance(block, IterativeToolOutputBlock):
            text = block.content.strip()
            if not text:
                continue
            fragments.append(fragment_from_iterative_tool_output_block(block))
            continue
        text = _block_text(block)
        if not text:
            continue
        source_id = _block_source_id(block, fallback=f"tool-{index}")
        meta: dict[str, Any] = {}
        if isinstance(block, dict):
            for key in ("tool_call_id", "tool_name", "step_id"):
                if key in block:
                    meta[key] = block[key]
        fragments.append(
            _fragment(
                fragment_id=f"tool-output-{source_id}",
                source=ContextFragmentSource.TOOL_OUTPUT,
                source_id=str(source_id),
                content=text,
                metadata=meta,
            )
        )
    return fragments


def fragments_from_system_instructions(
    instructions: str,
) -> list[ContextFragment]:
    """Emit mandatory SYSTEM_INSTRUCTIONS fragment."""
    text = (instructions or "").strip()
    if not text:
        return []
    return [
        _fragment(
            fragment_id="system-instructions",
            source=ContextFragmentSource.SYSTEM_INSTRUCTIONS,
            source_id="system",
            content=text,
            mandatory=True,
        )
    ]


def fragments_from_policy_overlay_fragments(
    overlays: list[Any],
) -> list[ContextFragment]:
    """Emit POLICY_OVERLAY fragments sorted by priority."""
    if not overlays:
        return []
    normalized: list[tuple[int, str, str]] = []
    for index, overlay in enumerate(overlays):
        if isinstance(overlay, dict):
            content = str(overlay.get("content") or "").strip()
            overlay_id = str(overlay.get("overlay_id") or f"overlay-{index}")
            priority = int(overlay.get("priority") or 100)
        else:
            content = str(attribute_access.optional(overlay, "content", "") or "").strip()
            overlay_id = str(attribute_access.optional(overlay, "overlay_id", None) or f"overlay-{index}")
            priority = int(attribute_access.optional(overlay, "priority", 100) or 100)
        if content:
            normalized.append((priority, overlay_id, content))
    normalized.sort(key=lambda row: row[0])
    return [
        _fragment(
            fragment_id=f"policy-{overlay_id}",
            source=ContextFragmentSource.POLICY_OVERLAY,
            source_id=overlay_id,
            content=content,
            mandatory=True,
            metadata={"priority": priority},
        )
        for priority, overlay_id, content in normalized
    ]


def fragments_from_attachment_summaries(
    summaries: list[Any],
    *,
    max_attachments: int = 8,
) -> list[ContextFragment]:
    """Emit ATTACHMENT fragments from ingestion summary rows."""
    if not summaries:
        return []
    fragments: list[ContextFragment] = []
    for index, summary in enumerate(summaries[:max_attachments]):
        text = _block_text(summary)
        if not text:
            continue
        attachment_id = _block_source_id(summary, fallback=f"attachment-{index}")
        meta: dict[str, Any] = {}
        if isinstance(summary, dict):
            for key in ("mime_type", "filename", "uri"):
                if key in summary:
                    meta[key] = summary[key]
        fragments.append(
            _fragment(
                fragment_id=f"attachment-{attachment_id}",
                source=ContextFragmentSource.ATTACHMENT,
                source_id=str(attachment_id),
                content=text,
                metadata=meta,
            )
        )
    return fragments


def fragments_from_shared_context_reads(
    reads: dict[str, Any],
) -> list[ContextFragment]:
    """Emit SHARED_CONTEXT fragments from precomputed graph shared reads."""
    if not reads:
        return []
    fragments: list[ContextFragment] = []
    for key, payload in reads.items():
        if key == "artifacts":
            if not payload:
                continue
            lines = []
            if isinstance(payload, dict):
                for label, artifact in payload.items():
                    lines.append(f"- {label}: {artifact}")
            content = "\n".join(lines).strip()
            if not content:
                continue
            fragments.append(
                _fragment(
                    fragment_id="shared-artifacts",
                    source=ContextFragmentSource.SHARED_CONTEXT,
                    source_id="artifacts",
                    content=content,
                    metadata={"shared_key": "artifacts"},
                )
            )
            continue
        if isinstance(payload, dict):
            content = str(payload.get("summary") or payload.get("content") or payload).strip()
        else:
            content = str(payload).strip()
        if not content:
            continue
        fragments.append(
            _fragment(
                fragment_id=f"shared-{key}",
                source=ContextFragmentSource.SHARED_CONTEXT,
                source_id=str(key),
                content=content,
                metadata={"shared_key": str(key)},
            )
        )
    return fragments
