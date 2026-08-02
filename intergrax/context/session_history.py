# © Artur Czarnecki. All rights reserved.

"""Structured session history contracts (CTX-UCL-3)."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from intergrax.context.contracts import (
    ContextAssemblyRequest,
    ContextFragment,
    ContextFragmentSource,
    ContextProviderContext,
    content_hash_for_text,
)
from intergrax.llm.messages import ChatMessage, MessageRole

SESSION_HISTORY_SNAPSHOT_HANDLE = "session_history_snapshot"


def _require_non_empty(value: str, field_name: str) -> str:
    if not value:
        raise ValueError(f"{field_name} must be non-empty")
    return value


def _require_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer")
    return value


def _normalize_tool_calls(
    tool_calls: Sequence[Mapping[str, Any]] | None,
) -> tuple[Mapping[str, Any], ...]:
    if not tool_calls:
        return ()
    normalized: list[Mapping[str, Any]] = []
    for item in tool_calls:
        if not isinstance(item, Mapping):
            raise ValueError("tool_calls items must be mappings")
        normalized.append(dict(item))
    return tuple(normalized)


def _tool_call_ids(tool_calls: tuple[Mapping[str, Any], ...]) -> tuple[str, ...]:
    ids: list[str] = []
    for call in tool_calls:
        call_id = call.get("id")
        if call_id is not None and str(call_id).strip():
            ids.append(str(call_id).strip())
    return tuple(ids)


def _message_content_hash(
    *,
    role: MessageRole,
    content: str,
    name: str | None,
    tool_call_id: str | None,
    tool_calls: tuple[Mapping[str, Any], ...],
) -> str:
    payload: dict[str, Any] = {
        "content": content,
        "name": name,
        "role": role,
        "tool_call_id": tool_call_id,
        "tool_calls": [dict(call) for call in tool_calls],
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class SessionHistoryMessage:
    message_id: str
    sequence: int
    role: MessageRole
    content: str
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: tuple[Mapping[str, Any], ...] = ()
    content_hash: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "message_id", _require_non_empty(self.message_id, "message_id"))
        sequence = _require_int(self.sequence, "sequence")
        if sequence < 0:
            raise ValueError("sequence must be >= 0")
        object.__setattr__(self, "sequence", sequence)
        if self.role not in {"system", "user", "assistant", "tool"}:
            raise ValueError("role must be one of system, user, assistant, tool")
        if not isinstance(self.content, str):
            raise ValueError("content must be a string")
        if self.name is not None:
            object.__setattr__(self, "name", _require_non_empty(self.name, "name"))
        if self.tool_call_id is not None:
            object.__setattr__(
                self,
                "tool_call_id",
                _require_non_empty(self.tool_call_id, "tool_call_id"),
            )
        normalized_calls = _normalize_tool_calls(self.tool_calls)
        object.__setattr__(self, "tool_calls", normalized_calls)
        if self.content_hash:
            object.__setattr__(
                self,
                "content_hash",
                _require_non_empty(self.content_hash, "content_hash"),
            )
        else:
            object.__setattr__(
                self,
                "content_hash",
                _message_content_hash(
                    role=self.role,
                    content=self.content,
                    name=self.name,
                    tool_call_id=self.tool_call_id,
                    tool_calls=normalized_calls,
                ),
            )

    @property
    def ordered_tool_call_ids(self) -> tuple[str, ...]:
        return _tool_call_ids(self.tool_calls)


def session_history_message_from_chat_message(
    message: ChatMessage,
    *,
    sequence: int,
) -> SessionHistoryMessage:
    entry_id = (message.entry_id or "").strip()
    if not entry_id:
        raise ValueError("ChatMessage.entry_id must be non-empty for session history")
    tool_calls = _normalize_tool_calls(message.tool_calls)
    return SessionHistoryMessage(
        message_id=entry_id,
        sequence=sequence,
        role=message.role,
        content=message.content or "",
        name=message.name,
        tool_call_id=message.tool_call_id,
        tool_calls=tool_calls,
    )


def session_history_message_to_chat_message(message: SessionHistoryMessage) -> ChatMessage:
    return ChatMessage(
        role=message.role,
        content=message.content,
        entry_id=message.message_id,
        name=message.name,
        tool_call_id=message.tool_call_id,
        tool_calls=[dict(call) for call in message.tool_calls] if message.tool_calls else None,
    )


def _snapshot_source_content_hash(messages: tuple[SessionHistoryMessage, ...]) -> str:
    rows: list[dict[str, Any]] = []
    for message in messages:
        rows.append(
            {
                "content_hash": message.content_hash,
                "message_id": message.message_id,
                "role": message.role,
                "sequence": message.sequence,
                "tool_call_id": message.tool_call_id,
                "tool_call_ids": list(message.ordered_tool_call_ids),
            }
        )
    canonical = json.dumps(rows, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class SessionHistorySnapshot:
    tenant_id: str
    context_scope_id: str
    revision_id: str
    messages: tuple[SessionHistoryMessage, ...]
    source_content_hash: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "tenant_id", _require_non_empty(self.tenant_id, "tenant_id"))
        object.__setattr__(
            self,
            "context_scope_id",
            _require_non_empty(self.context_scope_id, "context_scope_id"),
        )
        object.__setattr__(self, "revision_id", _require_non_empty(self.revision_id, "revision_id"))
        messages = tuple(self.messages)
        object.__setattr__(self, "messages", messages)

        seen_ids: set[str] = set()
        seen_sequences: set[int] = set()
        previous_sequence = -1
        for message in messages:
            if message.message_id in seen_ids:
                raise ValueError("message IDs must be unique")
            seen_ids.add(message.message_id)
            if message.sequence in seen_sequences:
                raise ValueError("sequences must be unique")
            seen_sequences.add(message.sequence)
            if message.sequence <= previous_sequence:
                raise ValueError("sequences must be strictly increasing")
            previous_sequence = message.sequence

        if self.source_content_hash:
            object.__setattr__(
                self,
                "source_content_hash",
                _require_non_empty(self.source_content_hash, "source_content_hash"),
            )
        else:
            object.__setattr__(self, "source_content_hash", _snapshot_source_content_hash(messages))

    @property
    def source_refs(self) -> tuple[str, ...]:
        return tuple(message.message_id for message in self.messages)


def build_session_history_snapshot(
    *,
    tenant_id: str,
    context_scope_id: str,
    revision_id: str,
    messages: Sequence[ChatMessage],
) -> SessionHistorySnapshot:
    """Build a complete session-history snapshot without slicing or synthetic IDs."""
    normalized: list[SessionHistoryMessage] = []
    seen_ids: set[str] = set()
    for sequence, message in enumerate(messages):
        entry_id = (message.entry_id or "").strip()
        if not entry_id:
            raise ValueError("ChatMessage.entry_id must be non-empty")
        if entry_id in seen_ids:
            raise ValueError("duplicate ChatMessage.entry_id in session history")
        seen_ids.add(entry_id)
        normalized.append(session_history_message_from_chat_message(message, sequence=sequence))
    return SessionHistorySnapshot(
        tenant_id=tenant_id,
        context_scope_id=context_scope_id,
        revision_id=revision_id,
        messages=tuple(normalized),
    )


def fragments_from_session_history_snapshot(
    snapshot: SessionHistorySnapshot,
) -> list[ContextFragment]:
    """Convert structured snapshot entries into CE fragments (canonical path)."""
    fragments: list[ContextFragment] = []
    for message in snapshot.messages:
        metadata: dict[str, Any] = {
            "message_id": message.message_id,
            "sequence": message.sequence,
            "role": message.role,
            "content_hash": message.content_hash,
            "context_scope_id": snapshot.context_scope_id,
            "revision_id": snapshot.revision_id,
        }
        if message.tool_call_id:
            metadata["tool_call_id"] = message.tool_call_id
        if message.ordered_tool_call_ids:
            metadata["tool_call_ids"] = list(message.ordered_tool_call_ids)
        fragments.append(
            ContextFragment(
                fragment_id=f"session-{message.message_id}",
                source=ContextFragmentSource.SESSION_HISTORY,
                source_id=message.message_id,
                content=message.content,
                token_estimate=max(1, len(message.content) // 4),
                relevance_score=0.75,
                freshness_score=0.7,
                confidence_score=0.8,
                mandatory=False,
                metadata=metadata,
                content_hash=message.content_hash,
            )
        )
    return fragments


class HandleSessionHistoryProvider:
    """Canonical handle-backed session history provider."""

    @property
    def provider_id(self) -> str:
        return "builtin.session_history_snapshot"

    async def load_snapshot(
        self,
        request: ContextAssemblyRequest,
        ctx: ContextProviderContext,
    ) -> SessionHistorySnapshot | None:
        if not request.decision_profile.include_session_history:
            return None
        raw = ctx.handles.get(SESSION_HISTORY_SNAPSHOT_HANDLE)
        if raw is None:
            return None
        if not isinstance(raw, SessionHistorySnapshot):
            raise ValueError(
                f"handle {SESSION_HISTORY_SNAPSHOT_HANDLE!r} must be SessionHistorySnapshot"
            )
        return raw

    async def collect(
        self,
        request: ContextAssemblyRequest,
        ctx: ContextProviderContext,
    ) -> list[ContextFragment]:
        snapshot = await self.load_snapshot(request, ctx)
        if snapshot is None:
            return []
        return fragments_from_session_history_snapshot(snapshot)
