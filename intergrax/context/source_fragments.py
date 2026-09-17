# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Typed source → ContextFragment adapters for canonical builtin providers."""

from __future__ import annotations

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
    fragments_from_session_history_snapshot,
    require_session_history_messages,
)
from intergrax.context.source_inputs import (
    ContextAttachmentSummaryInput,
    ContextMemoryEntryInput,
    ContextPolicyOverlayInput,
    ContextPriorOutputInput,
    ContextRagChunkInput,
    ContextSharedContextReadInput,
    ContextSystemInstructionsInput,
    ContextWebSearchResultInput,
)
from intergrax.llm.messages import ChatMessage

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
    messages: tuple[ChatMessage, ...] | None = None,
) -> list[ContextFragment]:
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


def fragments_from_memory_input(
    entries: tuple[ContextMemoryEntryInput, ...],
    *,
    max_entries: int,
) -> list[ContextFragment]:
    if not entries:
        return []
    fragments: list[ContextFragment] = []
    for index, entry in enumerate(entries[:max_entries]):
        if entry.deleted:
            continue
        text = entry.content.strip()
        if not text:
            continue
        meta: dict[str, Any] = {}
        if entry.kind is not None:
            meta["kind"] = entry.kind
        if entry.title is not None:
            meta["title"] = entry.title
        if entry.session_id is not None:
            meta["session_id"] = entry.session_id
        if entry.importance is not None:
            meta["importance"] = entry.importance
        if entry.raw_relevance_signal is not None:
            meta["raw_relevance_signal"] = entry.raw_relevance_signal
        fragments.append(
            _fragment(
                fragment_id=f"ltm-{entry.entry_id}",
                source=ContextFragmentSource.LONGTERM_MEMORY,
                source_id=entry.entry_id,
                content=text,
                metadata=meta,
            )
        )
    return fragments


def fragments_from_rag_input(
    chunks: tuple[ContextRagChunkInput, ...],
    *,
    max_chars: int = 4000,
) -> list[ContextFragment]:
    if not chunks:
        return []
    fragments: list[ContextFragment] = []
    total_chars = 0
    for index, chunk in enumerate(chunks):
        text = chunk.content.strip()
        if not text:
            continue
        if total_chars + len(text) > max_chars:
            remaining = max_chars - total_chars
            if remaining <= 80:
                break
            text = text[:remaining].rstrip() + "…"
        total_chars += len(text)
        citations = [{field.key: field.value} for field in chunk.citation_fields]
        meta: dict[str, Any] = {"citations": citations}
        for key, value in chunk.extra_metadata_keys:
            meta[key] = value
        if chunk.raw_relevance_signal is not None:
            meta["raw_relevance_signal"] = chunk.raw_relevance_signal
        fragments.append(
            _fragment(
                fragment_id=f"rag-{chunk.chunk_id}-{index}",
                source=ContextFragmentSource.RAG,
                source_id=chunk.chunk_id,
                content=text,
                metadata=meta,
            )
        )
        if total_chars >= max_chars:
            break
    return fragments


def fragments_from_web_input(
    blocks: tuple[ContextWebSearchResultInput, ...],
    *,
    max_blocks: int = 8,
) -> list[ContextFragment]:
    if not blocks:
        return []
    fragments: list[ContextFragment] = []
    for block in blocks[:max_blocks]:
        text = block.content.strip()
        if not text:
            continue
        meta: dict[str, Any] = {}
        if block.url is not None:
            meta["url"] = block.url
        if block.title is not None:
            meta["title"] = block.title
        if block.snippet is not None:
            meta["snippet"] = block.snippet
        fragments.append(
            _fragment(
                fragment_id=f"websearch-{block.source_id}",
                source=ContextFragmentSource.WEBSEARCH,
                source_id=block.source_id,
                content=text,
                metadata=meta,
            )
        )
    return fragments


def fragment_from_iterative_tool_output_block(block: IterativeToolOutputBlock) -> ContextFragment:
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


def fragments_from_tool_input(
    blocks: tuple[IterativeToolOutputBlock, ...],
    *,
    max_blocks: int = 16,
) -> list[ContextFragment]:
    fragments: list[ContextFragment] = []
    for block in blocks[:max_blocks]:
        text = block.content.strip()
        if not text:
            continue
        fragments.append(fragment_from_iterative_tool_output_block(block))
    return fragments


def fragments_from_system_input(
    instructions: ContextSystemInstructionsInput,
) -> list[ContextFragment]:
    text = instructions.text.strip()
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


def fragments_from_policy_overlay_input(
    overlays: tuple[ContextPolicyOverlayInput, ...],
) -> list[ContextFragment]:
    if not overlays:
        return []
    normalized = sorted(overlays, key=lambda row: row.priority)
    return [
        _fragment(
            fragment_id=f"policy-{overlay.overlay_id}",
            source=ContextFragmentSource.POLICY_OVERLAY,
            source_id=overlay.overlay_id,
            content=overlay.content.strip(),
            mandatory=True,
            metadata={"priority": overlay.priority},
        )
        for overlay in normalized
        if overlay.content.strip()
    ]


def fragments_from_attachment_input(
    summaries: tuple[ContextAttachmentSummaryInput, ...],
    *,
    max_attachments: int = 8,
) -> list[ContextFragment]:
    fragments: list[ContextFragment] = []
    for summary in summaries[:max_attachments]:
        text = summary.summary.strip()
        if not text:
            continue
        meta: dict[str, Any] = {}
        if summary.mime_type is not None:
            meta["mime_type"] = summary.mime_type
        if summary.filename is not None:
            meta["filename"] = summary.filename
        if summary.uri is not None:
            meta["uri"] = summary.uri
        fragments.append(
            _fragment(
                fragment_id=f"attachment-{summary.attachment_id}",
                source=ContextFragmentSource.ATTACHMENT,
                source_id=summary.attachment_id,
                content=text,
                metadata=meta,
            )
        )
    return fragments


def fragments_from_shared_context_input(
    reads: tuple[ContextSharedContextReadInput, ...],
) -> list[ContextFragment]:
    if not reads:
        return []
    fragments: list[ContextFragment] = []
    for read in reads:
        content = read.content.strip()
        if not content:
            continue
        fragments.append(
            _fragment(
                fragment_id=f"shared-{read.entry_key}",
                source=ContextFragmentSource.SHARED_CONTEXT,
                source_id=read.entry_key,
                content=content,
                metadata={"shared_key": read.entry_key},
            )
        )
    return fragments


def fragments_from_graph_prior_input(
    records: tuple[ContextPriorOutputInput, ...],
    *,
    max_entries: int | None = None,
) -> list[ContextFragment]:
    if not records:
        return []
    limit = max_entries if max_entries is not None else len(records)
    fragments: list[ContextFragment] = []
    for index, record in enumerate(records[:limit]):
        text = record.content.strip()
        if not text:
            continue
        meta: dict[str, Any] = {"node_id": record.node_id}
        if record.agent_id:
            meta["agent_id"] = record.agent_id
        fragments.append(
            _fragment(
                fragment_id=f"graph-prior-{record.node_id}-{index}",
                source=ContextFragmentSource.GRAPH_PRIOR,
                source_id=record.node_id,
                content=text,
                metadata=meta,
            )
        )
    return fragments


def fragments_from_legacy_session_messages(
    messages: list[ChatMessage],
    *,
    include_session_history: bool = True,
) -> list[ContextFragment]:
    if not include_session_history:
        return []
    if not messages:
        return []
    validated = require_session_history_messages(messages)
    if not validated:
        return []
    raise SessionHistorySnapshotRequiredError()


__all__ = [
    "fragments_from_attachment_input",
    "fragments_from_graph_prior_input",
    "fragments_from_memory_input",
    "fragments_from_policy_overlay_input",
    "fragments_from_rag_input",
    "fragments_from_shared_context_input",
    "fragments_from_system_input",
    "fragments_from_task_message",
    "fragments_from_tool_input",
    "fragments_from_web_input",
    "fragment_from_iterative_tool_output_block",
    "fragments_from_legacy_session_messages",
]
