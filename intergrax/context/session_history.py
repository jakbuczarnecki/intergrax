# © Artur Czarnecki. All rights reserved.

"""Structured session history contracts (CTX-UCL-3)."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from enum import Enum
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from intergrax.context.contracts import (
    BUILTIN_PROVIDER_VERSION,
    ContextAssemblyRequest,
    ContextFragment,
    ContextFragmentSource,
    ContextProviderContext,
    ContextProviderDescriptor,
)
from intergrax.llm.messages import ChatMessage, MessageRole

SESSION_HISTORY_SNAPSHOT_HANDLE = "session_history_snapshot"
SESSION_HISTORY_CONTEXT_SCOPE_HANDLE = "session_history_context_scope_id"
SESSION_HISTORY_REVISION_HANDLE = "session_history_revision_id"

SESSION_HISTORY_SNAPSHOT_REQUIRED_REASON = (
    "session_history_snapshot_required_for_ucl"
)

SESSION_HISTORY_SNAPSHOT_BINDING_REASON = (
    "session_history_snapshot_binding_mismatch"
)


class SessionHistorySnapshotRequiredError(RuntimeError):
    reason = SESSION_HISTORY_SNAPSHOT_REQUIRED_REASON

    def __init__(self) -> None:
        super().__init__(self.reason)


class SessionHistorySnapshotBindingError(RuntimeError):
    reason = SESSION_HISTORY_SNAPSHOT_BINDING_REASON

    def __init__(self) -> None:
        super().__init__(self.reason)


def _require_binding_non_empty_str(value: object) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise SessionHistorySnapshotBindingError()
    return value


def validate_session_history_snapshot_binding(
    snapshot: SessionHistorySnapshot,
    *,
    expected_tenant_id: str,
    expected_context_scope_id: str,
    expected_revision_id: str,
) -> SessionHistorySnapshot:
    if type(snapshot) is not SessionHistorySnapshot:
        raise SessionHistorySnapshotBindingError()
    tenant_id = _require_binding_non_empty_str(expected_tenant_id)
    context_scope_id = _require_binding_non_empty_str(expected_context_scope_id)
    revision_id = _require_binding_non_empty_str(expected_revision_id)
    if snapshot.tenant_id != tenant_id:
        raise SessionHistorySnapshotBindingError()
    if snapshot.context_scope_id != context_scope_id:
        raise SessionHistorySnapshotBindingError()
    if snapshot.revision_id != revision_id:
        raise SessionHistorySnapshotBindingError()
    return snapshot


def require_session_history_messages(
    raw: object,
    *,
    field_name: str = "session_history_messages",
) -> list[ChatMessage]:
    if raw is None:
        return []
    if raw == []:
        return []
    if type(raw) is not list:
        raise ValueError(f"{field_name} must be a list of ChatMessage")
    messages: list[ChatMessage] = []
    for item in raw:
        if type(item) is not ChatMessage:
            raise ValueError(f"{field_name} must contain only ChatMessage instances")
        messages.append(item)
    return messages


def _require_non_empty_str(
    value: object,
    field_name: str,
) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def _require_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer")
    return value


def _freeze_json_value(value: object) -> object:
    if value is None:
        return None
    if isinstance(value, Enum):
        raise ValueError(f"non-JSON-safe value: {type(value).__name__}")
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            raise ValueError("non-finite float not allowed")
        return value
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        frozen: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("mapping keys must be strings")
            frozen[key] = _freeze_json_value(item)
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json_value(item) for item in value)
    raise ValueError(f"non-JSON-safe value: {type(value).__name__}")


def _thaw_json_value(value: object) -> object:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, MappingProxyType) or isinstance(value, Mapping):
        return {str(key): _thaw_json_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json_value(item) for item in value]
    if isinstance(value, list):
        return [_thaw_json_value(item) for item in value]
    raise ValueError(f"cannot thaw value: {type(value).__name__}")


def _normalize_tool_calls(
    tool_calls: Sequence[Mapping[str, Any]] | None,
) -> tuple[Mapping[str, Any], ...]:
    if not tool_calls:
        return ()
    normalized: list[Mapping[str, Any]] = []
    for item in tool_calls:
        if not isinstance(item, Mapping):
            raise ValueError("tool_calls items must be mappings")
        normalized.append(_freeze_json_value(dict(item)))  # type: ignore[arg-type]
    return tuple(normalized)


def _tool_call_ids(tool_calls: tuple[Mapping[str, Any], ...]) -> tuple[str, ...]:
    ids: list[str] = []
    for call in tool_calls:
        call_id = call.get("id")
        if isinstance(call_id, str) and call_id.strip():
            ids.append(call_id.strip())
    return tuple(ids)


def _message_content_hash(
    *,
    role: MessageRole,
    content: str,
    name: str | None,
    tool_call_id: str | None,
    tool_calls: tuple[Mapping[str, Any], ...],
) -> str:
    thawed_calls = [_thaw_json_value(call) for call in tool_calls]
    payload: dict[str, Any] = {
        "content": content,
        "name": name,
        "role": role,
        "tool_call_id": tool_call_id,
        "tool_calls": thawed_calls,
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
        object.__setattr__(self, "message_id", _require_non_empty_str(self.message_id, "message_id"))
        sequence = _require_int(self.sequence, "sequence")
        if sequence < 0:
            raise ValueError("sequence must be >= 0")
        object.__setattr__(self, "sequence", sequence)
        if self.role not in {"system", "user", "assistant", "tool"}:
            raise ValueError("role must be one of system, user, assistant, tool")
        if not isinstance(self.content, str):
            raise ValueError("content must be a string")
        if self.name is not None:
            object.__setattr__(self, "name", _require_non_empty_str(self.name, "name"))
        if self.tool_call_id is not None:
            object.__setattr__(
                self,
                "tool_call_id",
                _require_non_empty_str(self.tool_call_id, "tool_call_id"),
            )
        normalized_calls = _normalize_tool_calls(self.tool_calls)
        object.__setattr__(self, "tool_calls", normalized_calls)
        expected_hash = _message_content_hash(
            role=self.role,
            content=self.content,
            name=self.name,
            tool_call_id=self.tool_call_id,
            tool_calls=normalized_calls,
        )
        if self.content_hash == "":
            object.__setattr__(self, "content_hash", expected_hash)
        else:
            stored_hash = _require_non_empty_str(self.content_hash, "content_hash")
            if stored_hash != expected_hash:
                raise ValueError("content_hash does not match canonical message content")
            object.__setattr__(self, "content_hash", stored_hash)

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
    thawed_calls = None
    if message.tool_calls:
        thawed_calls = [_thaw_json_value(call) for call in message.tool_calls]
    return ChatMessage(
        role=message.role,
        content=message.content,
        entry_id=message.message_id,
        name=message.name,
        tool_call_id=message.tool_call_id,
        tool_calls=thawed_calls,
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
        object.__setattr__(self, "tenant_id", _require_non_empty_str(self.tenant_id, "tenant_id"))
        object.__setattr__(
            self,
            "context_scope_id",
            _require_non_empty_str(self.context_scope_id, "context_scope_id"),
        )
        object.__setattr__(self, "revision_id", _require_non_empty_str(self.revision_id, "revision_id"))
        messages = tuple(self.messages)
        for message in messages:
            if not isinstance(message, SessionHistoryMessage):
                raise ValueError("messages must contain SessionHistoryMessage instances")
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

        expected_hash = _snapshot_source_content_hash(messages)
        if self.source_content_hash == "":
            object.__setattr__(self, "source_content_hash", expected_hash)
        else:
            stored_hash = _require_non_empty_str(self.source_content_hash, "source_content_hash")
            if stored_hash != expected_hash:
                raise ValueError("source_content_hash does not match snapshot messages")
            object.__setattr__(self, "source_content_hash", stored_hash)

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
        thawed_tool_calls = [_thaw_json_value(call) for call in message.tool_calls]
        metadata: dict[str, Any] = {
            "message_id": message.message_id,
            "sequence": message.sequence,
            "role": message.role,
            "name": message.name,
            "tool_call_id": message.tool_call_id,
            "tool_calls": thawed_tool_calls,
            "content_hash": message.content_hash,
            "context_scope_id": snapshot.context_scope_id,
            "revision_id": snapshot.revision_id,
        }
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


def session_history_chat_message_from_fragment(
    fragment: ContextFragment,
) -> ChatMessage:
    if fragment.source is not ContextFragmentSource.SESSION_HISTORY:
        raise ValueError("fragment must be session history")
    metadata = fragment.metadata
    message_id = metadata.get("message_id")
    if not isinstance(message_id, str) or not message_id.strip():
        raise ValueError("session history fragment metadata missing message_id")
    if fragment.source_id != message_id:
        raise ValueError("session history fragment source_id must match message_id")
    content_hash = metadata.get("content_hash")
    if not isinstance(content_hash, str) or fragment.content_hash != content_hash:
        raise ValueError("session history fragment content_hash mismatch")
    sequence = metadata.get("sequence")
    role = metadata.get("role")
    if role not in {"system", "user", "assistant", "tool"}:
        raise ValueError("session history fragment metadata missing role")
    name = metadata.get("name")
    if name is not None and (type(name) is not str or not name.strip()):
        raise ValueError("session history fragment metadata name is invalid")
    tool_call_id = metadata.get("tool_call_id")
    if tool_call_id is not None and (type(tool_call_id) is not str or not tool_call_id.strip()):
        raise ValueError("session history fragment metadata tool_call_id is invalid")
    raw_tool_calls = metadata.get("tool_calls", [])
    if raw_tool_calls is None:
        raw_tool_calls = []
    if type(raw_tool_calls) is not list:
        raise ValueError("session history fragment metadata tool_calls must be a list")
    tool_calls: list[dict[str, Any]] = []
    for item in raw_tool_calls:
        if not isinstance(item, dict):
            raise ValueError("session history fragment metadata tool_calls must be dict rows")
        tool_calls.append(dict(item))
    history_message = SessionHistoryMessage(
        message_id=message_id,
        sequence=_require_int(sequence, "sequence"),
        role=role,
        content=fragment.content,
        name=name,
        tool_call_id=tool_call_id,
        tool_calls=tuple(tool_calls),
        content_hash=content_hash,
    )
    return session_history_message_to_chat_message(history_message)


class HandleSessionHistoryProvider:
    """Canonical handle-backed session history provider."""

    _PROVIDER_VERSION = BUILTIN_PROVIDER_VERSION

    @property
    def provider_id(self) -> str:
        return "builtin.session_history_snapshot"

    @property
    def supported_sources(self) -> frozenset[ContextFragmentSource]:
        return frozenset({ContextFragmentSource.SESSION_HISTORY})

    @property
    def descriptor(self) -> ContextProviderDescriptor:
        return ContextProviderDescriptor(
            provider_id=self.provider_id,
            provider_version=self._PROVIDER_VERSION,
            supported_sources=self.supported_sources,
            origin="builtin",
        )

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
        expected_scope = ctx.handles.get(SESSION_HISTORY_CONTEXT_SCOPE_HANDLE)
        expected_revision = ctx.handles.get(SESSION_HISTORY_REVISION_HANDLE)
        if type(expected_scope) is not str or not expected_scope.strip():
            raise SessionHistorySnapshotBindingError()
        if type(expected_revision) is not str or not expected_revision.strip():
            raise SessionHistorySnapshotBindingError()
        return validate_session_history_snapshot_binding(
            raw,
            expected_tenant_id=request.tenant_id,
            expected_context_scope_id=expected_scope,
            expected_revision_id=expected_revision,
        )

    async def collect(
        self,
        request: ContextAssemblyRequest,
        ctx: ContextProviderContext,
    ) -> list[ContextFragment]:
        snapshot = await self.load_snapshot(request, ctx)
        if snapshot is None:
            return []
        return fragments_from_session_history_snapshot(snapshot)
