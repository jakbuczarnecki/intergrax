# © Artur Czarnecki. All rights reserved.

"""Legacy collector adapters for builtin context providers (CE-PROV-BRIDGE).

Handle key contract (``ContextProviderContext.handles``):

| Key | Type | Used by |
|-----|------|---------|
| ``prior_output_records`` | ``list[PriorOutputRecord]`` or dict rows | ``builtin.graph_prior`` |
| ``session_history_messages`` | ``list[ChatMessage]`` | ``builtin.session_history`` |
| ``messages`` | ``list[ChatMessage]`` | fallback for ``builtin.task_message`` |

``ContextAssemblyRequest.objective`` is the primary task-message source on graph/UAEP paths.
"""

from __future__ import annotations

from typing import Any

from intergrax.context.contracts import (
    ContextAssemblyRequest,
    ContextFragment,
    ContextFragmentSource,
    content_hash_for_text,
)
from intergrax.llm.messages import ChatMessage

PRIOR_OUTPUT_RECORDS_HANDLE = "prior_output_records"
SESSION_HISTORY_MESSAGES_HANDLE = "session_history_messages"

_DEFAULT_SCORES = {
    ContextFragmentSource.TASK_MESSAGE: (0.95, 0.9, 0.95),
    ContextFragmentSource.GRAPH_PRIOR: (0.85, 0.8, 0.85),
    ContextFragmentSource.SESSION_HISTORY: (0.75, 0.7, 0.8),
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
    if hasattr(record, "evidence") and (getattr(record, "evidence") or "").strip():
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
    """Emit SESSION_HISTORY fragments from chronological chat turns."""
    if not include_session_history or not messages:
        return []
    fragments: list[ContextFragment] = []
    for index, message in enumerate(messages[-max_entries:]):
        role = getattr(message, "role", None) or (
            message.get("role") if isinstance(message, dict) else "user"
        )
        content = getattr(message, "content", None)
        if content is None and isinstance(message, dict):
            content = message.get("content")
        text = str(content or "").strip()
        if not text:
            continue
        source_id = f"{role}-{index}"
        fragments.append(
            _fragment(
                fragment_id=f"session-{source_id}",
                source=ContextFragmentSource.SESSION_HISTORY,
                source_id=source_id,
                content=f"{role}: {text}",
                metadata={"role": str(role), "turn_index": index},
            )
        )
    return fragments
