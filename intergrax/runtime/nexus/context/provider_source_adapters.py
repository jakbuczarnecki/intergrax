# © Artur Czarnecki. All rights reserved.

"""Map runtime/task metadata rows into typed ``ContextProviderSourceInputs``."""

from __future__ import annotations

from typing import Any

from intergrax.context.contracts import IterativeToolOutputBlock
from intergrax.context.source_inputs import (
    ContextAttachmentSummaryInput,
    ContextMemoryEntryInput,
    ContextPolicyOverlayInput,
    ContextPriorOutputInput,
    ContextProviderSourceInputs,
    ContextRagChunkInput,
    ContextRagCitationField,
    ContextSessionSourceInput,
    ContextSharedContextReadInput,
    ContextSystemInstructionsInput,
    ContextWebSearchResultInput,
)
from intergrax.context.session_history import SessionHistorySnapshot
from intergrax.runtime.nexus.context.provider_handles import (
    ATTACHMENT_SUMMARIES_METADATA_KEY,
    LTM_ENTRIES_METADATA_KEY,
    POLICY_OVERLAY_FRAGMENTS_METADATA_KEY,
    RAG_CHUNKS_METADATA_KEY,
    SESSION_HISTORY_SNAPSHOT_METADATA_KEY,
    SYSTEM_INSTRUCTIONS_METADATA_KEY,
    TOOL_OUTPUT_BLOCKS_METADATA_KEY,
    WEBSEARCH_BLOCKS_METADATA_KEY,
)


def _str_field(row: dict[str, Any], key: str) -> str | None:
    value = row.get(key)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def memory_inputs_from_rows(rows: list[Any]) -> tuple[ContextMemoryEntryInput, ...]:
    result: list[ContextMemoryEntryInput] = []
    for index, row in enumerate(rows):
        if isinstance(row, ContextMemoryEntryInput):
            result.append(row)
            continue
        if not isinstance(row, dict):
            continue
        content = _str_field(row, "content") or _str_field(row, "text") or ""
        if not content:
            continue
        entry_id = _str_field(row, "entry_id") or _str_field(row, "id") or f"ltm-{index}"
        importance_raw = row.get("importance")
        importance = float(importance_raw) if isinstance(importance_raw, (int, float)) else None
        score_raw = row.get("score")
        signal = float(score_raw) if isinstance(score_raw, (int, float)) else None
        result.append(
            ContextMemoryEntryInput(
                entry_id=entry_id,
                content=content,
                kind=_str_field(row, "kind"),
                title=_str_field(row, "title"),
                session_id=_str_field(row, "session_id"),
                importance=importance,
                deleted=bool(row.get("deleted")),
                raw_relevance_signal=signal,
            )
        )
    return tuple(result)


def rag_inputs_from_rows(rows: list[Any]) -> tuple[ContextRagChunkInput, ...]:
    result: list[ContextRagChunkInput] = []
    for index, row in enumerate(rows):
        if isinstance(row, ContextRagChunkInput):
            result.append(row)
            continue
        if isinstance(row, dict):
            text = _str_field(row, "text") or _str_field(row, "content") or ""
            meta = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
            if not text and isinstance(meta, dict):
                text = _str_field(meta, "text") or _str_field(meta, "content") or ""
            if not text:
                continue
            chunk_id = (
                _str_field(meta, "chunk_id")
                or _str_field(meta, "doc_id")
                or _str_field(meta, "id")
                or f"rag-{index}"
            )
            citations: list[ContextRagCitationField] = []
            if isinstance(meta, dict):
                for key in ("source", "url", "doc_id", "file", "page", "page_number"):
                    if meta.get(key) is not None:
                        citations.append(ContextRagCitationField(key=key, value=str(meta[key])))
            extra = tuple(
                (str(k), str(v))
                for k, v in meta.items()
                if k not in {"source", "url", "doc_id", "file", "page", "page_number", "id", "chunk_id", "text", "content"}
            )
            score_raw = meta.get("score") if isinstance(meta, dict) else None
            signal = float(score_raw) if isinstance(score_raw, (int, float)) else None
            result.append(
                ContextRagChunkInput(
                    chunk_id=chunk_id,
                    content=text,
                    citation_fields=tuple(citations),
                    extra_metadata_keys=extra,
                    raw_relevance_signal=signal,
                )
            )
    return tuple(result)


def web_inputs_from_rows(rows: list[Any]) -> tuple[ContextWebSearchResultInput, ...]:
    result: list[ContextWebSearchResultInput] = []
    for index, row in enumerate(rows):
        if isinstance(row, ContextWebSearchResultInput):
            result.append(row)
            continue
        if isinstance(row, str):
            text = row.strip()
            if text:
                result.append(ContextWebSearchResultInput(source_id=f"web-{index}", content=text))
            continue
        if isinstance(row, dict):
            content = _str_field(row, "content") or _str_field(row, "text") or _str_field(row, "summary") or ""
            if not content:
                continue
            source_id = _str_field(row, "source_id") or _str_field(row, "id") or f"web-{index}"
            result.append(
                ContextWebSearchResultInput(
                    source_id=source_id,
                    content=content,
                    url=_str_field(row, "url"),
                    title=_str_field(row, "title"),
                    snippet=_str_field(row, "snippet"),
                )
            )
    return tuple(result)


def tool_inputs_from_rows(rows: list[Any]) -> tuple[IterativeToolOutputBlock, ...]:
    result: list[IterativeToolOutputBlock] = []
    for index, row in enumerate(rows):
        if isinstance(row, IterativeToolOutputBlock):
            result.append(row)
            continue
        if isinstance(row, dict):
            content = _str_field(row, "content") or _str_field(row, "text") or ""
            if not content:
                continue
            tool_call_id = _str_field(row, "tool_call_id") or f"tool-{index}"
            tool_name = _str_field(row, "tool_name") or "tool"
            step_id = _str_field(row, "step_id")
            result.append(
                IterativeToolOutputBlock(
                    content=content,
                    tool_call_id=tool_call_id,
                    tool_name=tool_name,
                    step_id=step_id,
                )
            )
    return tuple(result)


def attachment_inputs_from_rows(rows: list[Any]) -> tuple[ContextAttachmentSummaryInput, ...]:
    result: list[ContextAttachmentSummaryInput] = []
    for index, row in enumerate(rows):
        if isinstance(row, ContextAttachmentSummaryInput):
            result.append(row)
            continue
        if isinstance(row, dict):
            summary = _str_field(row, "summary") or _str_field(row, "content") or ""
            if not summary:
                continue
            attachment_id = _str_field(row, "attachment_id") or f"attachment-{index}"
            result.append(
                ContextAttachmentSummaryInput(
                    attachment_id=attachment_id,
                    summary=summary,
                    mime_type=_str_field(row, "mime_type"),
                    filename=_str_field(row, "filename"),
                    uri=_str_field(row, "uri"),
                )
            )
    return tuple(result)


def policy_overlay_inputs_from_rows(rows: list[Any]) -> tuple[ContextPolicyOverlayInput, ...]:
    result: list[ContextPolicyOverlayInput] = []
    for index, row in enumerate(rows):
        if isinstance(row, ContextPolicyOverlayInput):
            result.append(row)
            continue
        if isinstance(row, dict):
            content = _str_field(row, "content") or ""
            if not content:
                continue
            overlay_id = _str_field(row, "overlay_id") or f"overlay-{index}"
            priority_raw = row.get("priority")
            priority = int(priority_raw) if isinstance(priority_raw, int) else 100
            result.append(ContextPolicyOverlayInput(overlay_id=overlay_id, content=content, priority=priority))
    return tuple(result)


def shared_context_inputs_from_mapping(
    reads: dict[str, Any],
) -> tuple[ContextSharedContextReadInput, ...]:
    if not reads:
        return ()
    entries: list[ContextSharedContextReadInput] = []
    for key, payload in reads.items():
        if key == "artifacts":
            if not payload:
                continue
            lines: list[str] = []
            if isinstance(payload, dict):
                for label, artifact in payload.items():
                    lines.append(f"- {label}: {artifact}")
            content = "\n".join(lines).strip()
            if content:
                entries.append(ContextSharedContextReadInput(entry_key="artifacts", content=content))
            continue
        if isinstance(payload, dict):
            content = str(payload.get("summary") or payload.get("content") or payload).strip()
        else:
            content = str(payload).strip()
        if content:
            entries.append(ContextSharedContextReadInput(entry_key=str(key), content=content))
    return tuple(entries)


def prior_output_inputs_from_records(records: list[Any]) -> tuple[ContextPriorOutputInput, ...]:
    result: list[ContextPriorOutputInput] = []
    for record in records:
        if isinstance(record, ContextPriorOutputInput):
            result.append(record)
            continue
        evidence = ""
        summary = ""
        node_id = "unknown"
        agent_id: str | None = None
        if isinstance(record, dict):
            evidence = str(record.get("evidence") or "").strip()
            summary = str(record.get("summary") or "").strip()
            node_id = str(record.get("node_id") or "unknown")
            agent_id = _str_field(record, "agent_id")
        else:
            from intergrax.runtime.nexus.context.context_models import PriorOutputRecord

            if isinstance(record, PriorOutputRecord):
                evidence = str(record.evidence or "").strip()
                summary = str(record.summary or "").strip()
                node_id = str(record.node_id)
                agent_id = str(record.agent_id or "").strip() or None
            else:
                continue
        content = evidence or summary
        if not content:
            continue
        result.append(ContextPriorOutputInput(node_id=node_id, content=content, agent_id=agent_id))
    return tuple(result)


def sources_from_task_metadata(
    task_metadata: dict[str, Any],
    *,
    prior_output_records: list[Any] | None = None,
    shared_context_reads: dict[str, Any] | None = None,
    session_snapshot: SessionHistorySnapshot | None = None,
) -> ContextProviderSourceInputs:
    """Build typed provider sources from graph/task metadata (canonical staging)."""
    memory_rows = task_metadata.get(LTM_ENTRIES_METADATA_KEY)
    rag_rows = task_metadata.get(RAG_CHUNKS_METADATA_KEY)
    web_rows = task_metadata.get(WEBSEARCH_BLOCKS_METADATA_KEY)
    tool_rows = task_metadata.get(TOOL_OUTPUT_BLOCKS_METADATA_KEY)
    system_text = task_metadata.get(SYSTEM_INSTRUCTIONS_METADATA_KEY)
    policy_rows = task_metadata.get(POLICY_OVERLAY_FRAGMENTS_METADATA_KEY)
    attachment_rows = task_metadata.get(ATTACHMENT_SUMMARIES_METADATA_KEY)
    direct_snapshot = task_metadata.get(SESSION_HISTORY_SNAPSHOT_METADATA_KEY)

    session_input: ContextSessionSourceInput | None = None
    resolved_snapshot = session_snapshot
    if resolved_snapshot is None and type(direct_snapshot) is SessionHistorySnapshot:
        resolved_snapshot = direct_snapshot
    if resolved_snapshot is not None:
        session_input = ContextSessionSourceInput(
            snapshot=resolved_snapshot,
            binding_context_scope_id=resolved_snapshot.context_scope_id,
            binding_revision_id=resolved_snapshot.revision_id,
        )

    system_input: ContextSystemInstructionsInput | None = None
    if isinstance(system_text, str) and system_text.strip():
        system_input = ContextSystemInstructionsInput(text=system_text.strip())

    return ContextProviderSourceInputs(
        memory=memory_inputs_from_rows(list(memory_rows)) if isinstance(memory_rows, list) else (),
        rag=rag_inputs_from_rows(list(rag_rows)) if isinstance(rag_rows, list) else (),
        web=web_inputs_from_rows(list(web_rows)) if isinstance(web_rows, list) else (),
        tools=tool_inputs_from_rows(list(tool_rows)) if isinstance(tool_rows, list) else (),
        session=session_input,
        system=system_input,
        attachments=attachment_inputs_from_rows(list(attachment_rows))
        if isinstance(attachment_rows, list)
        else (),
        policy_overlay=policy_overlay_inputs_from_rows(list(policy_rows)) if isinstance(policy_rows, list) else (),
        shared_context=shared_context_inputs_from_mapping(shared_context_reads or {}),
        graph_prior=prior_output_inputs_from_records(list(prior_output_records or ())),
    )
