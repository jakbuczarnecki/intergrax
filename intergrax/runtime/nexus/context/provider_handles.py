# © Artur Czarnecki. All rights reserved.

"""Provider handle builders for graph context assembly (CE-PROV-CTX, CE-PROV-WIRE)."""

from __future__ import annotations

from typing import Any

from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.task.task import Task

from intergrax.context.session_history import SessionHistorySnapshotRequiredError

SESSION_HISTORY_SNAPSHOT_METADATA_KEY = "session_history_snapshot"
SESSION_CONTEXT_REVISION_METADATA_KEY = "session_context_revision_id"

WORKSPACE_FILES_METADATA_KEY = "workspace_files"
SESSION_HISTORY_MESSAGES_METADATA_KEY = "session_history_messages"
RAG_CHUNKS_METADATA_KEY = "rag_chunks"
LTM_ENTRIES_METADATA_KEY = "ltm_entries"
WEBSEARCH_BLOCKS_METADATA_KEY = "websearch_blocks"
TOOL_OUTPUT_BLOCKS_METADATA_KEY = "tool_output_blocks"
SYSTEM_INSTRUCTIONS_METADATA_KEY = "system_instructions"
POLICY_OVERLAY_FRAGMENTS_METADATA_KEY = "policy_overlay_fragments"
ATTACHMENT_SUMMARIES_METADATA_KEY = "attachment_summaries"


def workspace_files_from_task(task: Task) -> dict[str, str]:
    """Read workspace file map from task metadata when present."""
    raw = task.metadata.get(WORKSPACE_FILES_METADATA_KEY)
    if not isinstance(raw, dict) or not raw:
        return {}
    return {str(path): str(content) for path, content in raw.items()}


def session_history_messages_from_task(task: Task) -> list[Any]:
    """Read session history turns from task metadata when present."""
    from intergrax.context.session_history import require_session_history_messages

    return require_session_history_messages(
        task.metadata.get(SESSION_HISTORY_MESSAGES_METADATA_KEY),
        field_name="session_history_messages",
    )


def _require_non_empty_task_string(value: object) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise SessionHistorySnapshotRequiredError()
    return value


def _session_history_binding_inputs(task: Task) -> tuple[str, str, str]:
    tenant_id = _require_non_empty_task_string(task.tenant_id)
    session_id = _require_non_empty_task_string(task.session_id)
    revision_id = _require_non_empty_task_string(
        task.metadata.get(SESSION_CONTEXT_REVISION_METADATA_KEY)
    )
    return tenant_id, session_id, revision_id


def try_build_session_history_snapshot(
    task: Task,
    messages: list[Any],
) -> object | None:
    """Build a canonical snapshot when stable scope and revision identifiers exist."""
    revision_id = str(task.metadata.get(SESSION_CONTEXT_REVISION_METADATA_KEY) or "").strip()
    if not revision_id:
        return None
    return try_build_session_history_snapshot_from_scope(
        tenant_id=task.tenant_id,
        context_scope_id=task.session_id,
        revision_id=revision_id,
        messages=messages,
    )


def try_build_session_history_snapshot_from_scope(
    *,
    tenant_id: str,
    context_scope_id: str | None,
    revision_id: str | None,
    messages: list[Any],
) -> object | None:
    from intergrax.context.session_history import build_session_history_snapshot
    from intergrax.llm.messages import ChatMessage

    if not messages:
        return None
    scope = (context_scope_id or "").strip()
    if not scope:
        return None
    resolved_revision = (revision_id or "").strip()
    if not resolved_revision:
        return None
    tenant = (tenant_id or "").strip()
    if not tenant:
        return None
    typed_messages: list[ChatMessage] = []
    for item in messages:
        if isinstance(item, ChatMessage):
            typed_messages.append(item)
        else:
            raise ValueError("messages must be ChatMessage instances")
    return build_session_history_snapshot(
        tenant_id=tenant,
        context_scope_id=scope,
        revision_id=resolved_revision,
        messages=typed_messages,
    )


def _list_from_task_metadata(task: Task, key: str) -> list[Any]:
    raw = task.metadata.get(key)
    if not isinstance(raw, list) or not raw:
        return []
    return list(raw)


def _str_from_task_metadata(task: Task, key: str) -> str | None:
    raw = task.metadata.get(key)
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    return None


def build_graph_provider_handles(
    task: Task,
    *,
    runtime_config: RuntimeConfig,
    messages: list[Any],
    event_bus: RuntimeEventBus | None,
    node_id: str,
    agent_id: str | None,
    engine_id: str,
    prior_output_records: list[Any] | None = None,
    session_history_messages: list[Any] | None = None,
    shared_context_reads: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble runtime handles for ``ContextProviderContext`` on graph nodes."""
    from intergrax.context.providers.legacy_bridge import (
        ATTACHMENT_SUMMARIES_HANDLE,
        LTM_ENTRIES_HANDLE,
        POLICY_OVERLAY_FRAGMENTS_HANDLE,
        PRIOR_OUTPUT_RECORDS_HANDLE,
        RAG_CHUNKS_HANDLE,
        SHARED_CONTEXT_READS_HANDLE,
        SYSTEM_INSTRUCTIONS_HANDLE,
        TOOL_OUTPUT_BLOCKS_HANDLE,
        WEBSEARCH_BLOCKS_HANDLE,
    )
    from intergrax.context.session_history import (
        SESSION_HISTORY_CONTEXT_SCOPE_HANDLE,
        SESSION_HISTORY_REVISION_HANDLE,
        SESSION_HISTORY_SNAPSHOT_HANDLE,
        SessionHistorySnapshot,
        build_session_history_snapshot,
        require_session_history_messages,
        validate_session_history_snapshot_binding,
    )

    handles: dict[str, Any] = {
        "runtime_config": runtime_config,
        "messages": messages,
        "event_bus": event_bus,
        "node_id": node_id,
        "agent_id": agent_id,
        "engine_id": engine_id,
    }
    workspace_files = workspace_files_from_task(task)
    if workspace_files:
        handles["workspace_files"] = workspace_files
    memory_profile = task.metadata.get("memory_profile")
    if isinstance(memory_profile, dict) and memory_profile.get("enable_session_vector_index"):
        handles["enable_session_vector_index"] = True
    vector_hits = task.metadata.get("session_vector_hits")
    if isinstance(vector_hits, list):
        handles["session_vector_hits"] = vector_hits
    if prior_output_records:
        handles[PRIOR_OUTPUT_RECORDS_HANDLE] = list(prior_output_records)
    direct_snapshot = task.metadata.get(SESSION_HISTORY_SNAPSHOT_METADATA_KEY)
    if direct_snapshot is not None:
        if type(direct_snapshot) is not SessionHistorySnapshot:
            raise ValueError("session_history_snapshot must be SessionHistorySnapshot")
        tenant_id, session_id, revision_id = _session_history_binding_inputs(task)
        validate_session_history_snapshot_binding(
            direct_snapshot,
            expected_tenant_id=tenant_id,
            expected_context_scope_id=session_id,
            expected_revision_id=revision_id,
        )
        handles[SESSION_HISTORY_SNAPSHOT_HANDLE] = direct_snapshot
        handles[SESSION_HISTORY_CONTEXT_SCOPE_HANDLE] = session_id
        handles[SESSION_HISTORY_REVISION_HANDLE] = revision_id
    else:
        if session_history_messages is None:
            history_messages = require_session_history_messages(
                task.metadata.get(SESSION_HISTORY_MESSAGES_METADATA_KEY),
                field_name="session_history_messages",
            )
        else:
            history_messages = require_session_history_messages(session_history_messages)
        if history_messages:
            tenant_id, session_id, revision_id = _session_history_binding_inputs(task)
            snapshot = build_session_history_snapshot(
                tenant_id=tenant_id,
                context_scope_id=session_id,
                revision_id=revision_id,
                messages=history_messages,
            )
            validate_session_history_snapshot_binding(
                snapshot,
                expected_tenant_id=tenant_id,
                expected_context_scope_id=session_id,
                expected_revision_id=revision_id,
            )
            handles[SESSION_HISTORY_SNAPSHOT_HANDLE] = snapshot
            handles[SESSION_HISTORY_CONTEXT_SCOPE_HANDLE] = session_id
            handles[SESSION_HISTORY_REVISION_HANDLE] = revision_id
    rag_chunks = _list_from_task_metadata(task, RAG_CHUNKS_METADATA_KEY)
    if rag_chunks:
        handles[RAG_CHUNKS_HANDLE] = rag_chunks
    ltm_entries = _list_from_task_metadata(task, LTM_ENTRIES_METADATA_KEY)
    if ltm_entries:
        handles[LTM_ENTRIES_HANDLE] = ltm_entries
    websearch_blocks = _list_from_task_metadata(task, WEBSEARCH_BLOCKS_METADATA_KEY)
    if websearch_blocks:
        handles[WEBSEARCH_BLOCKS_HANDLE] = websearch_blocks
    tool_output_blocks = _list_from_task_metadata(task, TOOL_OUTPUT_BLOCKS_METADATA_KEY)
    if tool_output_blocks:
        handles[TOOL_OUTPUT_BLOCKS_HANDLE] = tool_output_blocks
    system_instructions = _str_from_task_metadata(task, SYSTEM_INSTRUCTIONS_METADATA_KEY)
    if system_instructions:
        handles[SYSTEM_INSTRUCTIONS_HANDLE] = system_instructions
    policy_overlays = _list_from_task_metadata(task, POLICY_OVERLAY_FRAGMENTS_METADATA_KEY)
    if policy_overlays:
        handles[POLICY_OVERLAY_FRAGMENTS_HANDLE] = policy_overlays
    attachment_summaries = _list_from_task_metadata(task, ATTACHMENT_SUMMARIES_METADATA_KEY)
    if attachment_summaries:
        handles[ATTACHMENT_SUMMARIES_HANDLE] = attachment_summaries
    if shared_context_reads:
        handles[SHARED_CONTEXT_READS_HANDLE] = dict(shared_context_reads)
    return handles
