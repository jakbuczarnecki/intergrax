# © Artur Czarnecki. All rights reserved.

"""Bounded Conversation thread memory adapter over SessionHistorySnapshot (LKW-CONVERSATION-CONTEXT-1C)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Protocol
from uuid import uuid4

from intergrax.context.session_history import (
    SessionHistoryMessage,
    SessionHistorySnapshot,
    session_history_message_from_chat_message,
)
from intergrax.llm.messages import ChatMessage
from intergrax.integrations.contracts.document_store import (
    ConditionalDocumentStore,
    DocumentRecord,
    DocumentStore,
)
from local_workspace_application.workspaces.conversation_context_models import (
    ConversationAudienceMode,
    ConversationExecutionContextV1,
    ConversationThreadMemoryLimitsV1,
    ConversationThreadMemoryMessageRole,
    ConversationThreadMemoryMessageV1,
)

_PARTITION_SCHEMA_VERSION = 1
_SESSION_KEY_PREFIX = "lkw-conversation-thread:v1:"
_REVISION_ID_PREFIX = "lkw-thread-revision:v1:"
_DOCUMENT_PARTITION = "lkw.conversation_thread_memory:v1"
_DOCUMENT_TTL_SECONDS = 7 * 24 * 60 * 60
_MAX_APPLIED_EXCHANGE_IDS = 64


class ConversationThreadMemoryError(RuntimeError):
    def __init__(self, error_code: str) -> None:
        super().__init__(error_code)
        self.error_code = error_code


@dataclass(frozen=True, slots=True)
class ConversationThreadMemoryPartitionV1:
    tenant_id: str
    conversation_context_binding_id: str
    canonical_thread_ref: str
    audience_mode: ConversationAudienceMode
    workspace_id: str

    @classmethod
    def from_execution_context(
        cls,
        context: ConversationExecutionContextV1,
    ) -> ConversationThreadMemoryPartitionV1:
        return cls(
            tenant_id=context.tenant_id,
            conversation_context_binding_id=context.conversation_context_binding_id,
            canonical_thread_ref=context.canonical_thread_ref,
            audience_mode=context.audience_mode,
            workspace_id=context.workspace_id,
        )


def derive_conversation_thread_session_key(
    *,
    tenant_id: str,
    conversation_context_binding_id: str,
    canonical_thread_ref: str,
) -> str:
    payload = {
        "version": 1,
        "tenant_id": tenant_id,
        "conversation_context_binding_id": conversation_context_binding_id,
        "canonical_thread_ref": canonical_thread_ref,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"{_SESSION_KEY_PREFIX}{digest}"


@dataclass(frozen=True, slots=True)
class ConversationThreadMemorySnapshotV1:
    """Adapter-owned immutable envelope carrying a platform SessionHistorySnapshot."""

    schema_version: int
    tenant_id: str
    conversation_context_binding_id: str
    canonical_thread_ref: str
    audience_mode: ConversationAudienceMode
    workspace_id: str
    snapshot: SessionHistorySnapshot
    message_created_at: tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class ThreadMemoryLifecycleEnvelopeV1:
    memory_snapshot: ConversationThreadMemorySnapshotV1
    applied_exchange_ids: tuple[str, ...] = ()

    @property
    def revision_id(self) -> str:
        return self.memory_snapshot.snapshot.revision_id

    def __post_init__(self) -> None:
        normalized = tuple(item.strip() for item in self.applied_exchange_ids)
        if (
            len(normalized) > _MAX_APPLIED_EXCHANGE_IDS
            or any(not item for item in normalized)
            or len(normalized) != len(set(normalized))
        ):
            raise ValueError("invalid applied exchange identity list")
        object.__setattr__(self, "applied_exchange_ids", normalized)


class ThreadMemoryLifecyclePort(Protocol):
    """Atomic compare-and-set lifecycle port for thread memory envelopes.

    Implementations must perform the revision comparison and envelope
    replacement as one atomic storage operation.

    ``save_envelope`` semantics:

    * ``expected_revision_id is None`` — create only when no envelope exists.
    * ``expected_revision_id`` is a revision — replace only when the currently
      stored revision exactly matches it.
    * Successful create or replace — return ``True``.
    * Absent envelope or current revision mismatch — return ``False``.
    """

    def load_envelope(
        self,
        *,
        partition: ConversationThreadMemoryPartitionV1,
    ) -> ThreadMemoryLifecycleEnvelopeV1 | None:
        ...

    def save_envelope(
        self,
        *,
        partition: ConversationThreadMemoryPartitionV1,
        envelope: ThreadMemoryLifecycleEnvelopeV1,
        expected_revision_id: str | None,
    ) -> bool:
        ...


def _serialize_envelope(envelope: ThreadMemoryLifecycleEnvelopeV1) -> dict[str, Any]:
    memory_snapshot = envelope.memory_snapshot
    return {
        "memory_snapshot": {
            "schema_version": memory_snapshot.schema_version,
            "tenant_id": memory_snapshot.tenant_id,
            "conversation_context_binding_id": memory_snapshot.conversation_context_binding_id,
            "canonical_thread_ref": memory_snapshot.canonical_thread_ref,
            "audience_mode": memory_snapshot.audience_mode.value,
            "workspace_id": memory_snapshot.workspace_id,
            "snapshot": {
                "tenant_id": memory_snapshot.snapshot.tenant_id,
                "context_scope_id": memory_snapshot.snapshot.context_scope_id,
                "revision_id": memory_snapshot.snapshot.revision_id,
                "messages": [
                    {
                        "message_id": message.message_id,
                        "sequence": message.sequence,
                        "role": message.role,
                        "content": message.content,
                        "name": message.name,
                        "tool_call_id": message.tool_call_id,
                        "tool_calls": list(message.tool_calls),
                        "content_hash": message.content_hash,
                    }
                    for message in memory_snapshot.snapshot.messages
                ],
                "source_content_hash": memory_snapshot.snapshot.source_content_hash,
            },
            "message_created_at": [list(item) for item in memory_snapshot.message_created_at],
        },
        "applied_exchange_ids": list(envelope.applied_exchange_ids),
    }


def _deserialize_envelope(data: object) -> ThreadMemoryLifecycleEnvelopeV1:
    if not isinstance(data, dict):
        raise ConversationThreadMemoryError("THREAD_MEMORY_ENVELOPE_MALFORMED")
    raw_snapshot = data.get("memory_snapshot")
    if not isinstance(raw_snapshot, dict):
        raise ConversationThreadMemoryError("THREAD_MEMORY_ENVELOPE_MALFORMED")
    raw_platform_snapshot = raw_snapshot.get("snapshot")
    raw_messages = (
        raw_platform_snapshot.get("messages")
        if isinstance(raw_platform_snapshot, dict)
        else None
    )
    if not isinstance(raw_platform_snapshot, dict) or not isinstance(raw_messages, list):
        raise ConversationThreadMemoryError("THREAD_MEMORY_ENVELOPE_MALFORMED")
    try:
        platform_messages = tuple(
            SessionHistoryMessage(
                **{
                    **dict(message),
                    "tool_calls": tuple(dict(call) for call in message.get("tool_calls", ())),
                }
            )
            for message in raw_messages
            if isinstance(message, dict)
        )
        if len(platform_messages) != len(raw_messages):
            raise ValueError("message must be an object")
        snapshot = SessionHistorySnapshot(
            tenant_id=raw_platform_snapshot["tenant_id"],
            context_scope_id=raw_platform_snapshot["context_scope_id"],
            revision_id=raw_platform_snapshot["revision_id"],
            messages=platform_messages,
            source_content_hash=raw_platform_snapshot.get("source_content_hash", ""),
        )
        created_at = tuple(
            (str(item[0]), str(item[1]))
            for item in raw_snapshot["message_created_at"]
        )
        memory_snapshot = ConversationThreadMemorySnapshotV1(
            schema_version=int(raw_snapshot["schema_version"]),
            tenant_id=str(raw_snapshot["tenant_id"]),
            conversation_context_binding_id=str(
                raw_snapshot["conversation_context_binding_id"]
            ),
            canonical_thread_ref=str(raw_snapshot["canonical_thread_ref"]),
            audience_mode=ConversationAudienceMode(raw_snapshot["audience_mode"]),
            workspace_id=str(raw_snapshot["workspace_id"]),
            snapshot=snapshot,
            message_created_at=created_at,
        )
        return ThreadMemoryLifecycleEnvelopeV1(
            memory_snapshot=memory_snapshot,
            applied_exchange_ids=tuple(
                str(item) for item in data.get("applied_exchange_ids", ())
            ),
        )
    except ConversationThreadMemoryError:
        raise
    except Exception as exc:  # noqa: BLE001 - normalized storage boundary
        raise ConversationThreadMemoryError("THREAD_MEMORY_ENVELOPE_MALFORMED") from exc


class DocumentStoreThreadMemoryLifecyclePort:
    """Durable atomic thread-memory lifecycle over the shared DocumentStore."""

    def __init__(
        self,
        document_store: DocumentStore,
        *,
        ttl_seconds: int = _DOCUMENT_TTL_SECONDS,
    ) -> None:
        if isinstance(ttl_seconds, bool) or ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        self._store = document_store
        self._ttl_seconds = ttl_seconds

    def _conditional_store(self) -> ConditionalDocumentStore:
        if not isinstance(self._store, ConditionalDocumentStore):
            raise ConversationThreadMemoryError(
                "THREAD_MEMORY_CONDITIONAL_STORE_REQUIRED"
            )
        return self._store

    @staticmethod
    def _row_key(partition: ConversationThreadMemoryPartitionV1) -> str:
        return derive_conversation_thread_session_key(
            tenant_id=partition.tenant_id,
            conversation_context_binding_id=partition.conversation_context_binding_id,
            canonical_thread_ref=partition.canonical_thread_ref,
        )

    def _record(
        self,
        *,
        partition: ConversationThreadMemoryPartitionV1,
        envelope: ThreadMemoryLifecycleEnvelopeV1,
    ) -> DocumentRecord:
        return DocumentRecord(
            partition_key=_DOCUMENT_PARTITION,
            row_key=self._row_key(partition),
            data=_serialize_envelope(envelope),
            ttl_seconds=self._ttl_seconds,
        )

    def load_envelope(
        self,
        *,
        partition: ConversationThreadMemoryPartitionV1,
    ) -> ThreadMemoryLifecycleEnvelopeV1 | None:
        stored = self._store.get(_DOCUMENT_PARTITION, self._row_key(partition))
        if stored is None:
            return None
        try:
            envelope = _deserialize_envelope(dict(stored.data))
            _validate_snapshot_partition(
                memory_snapshot=envelope.memory_snapshot,
                partition=partition,
            )
            _validate_snapshot_timestamps(envelope.memory_snapshot)
            return envelope
        except ConversationThreadMemoryError:
            raise
        except Exception as exc:  # noqa: BLE001 - normalized storage boundary
            raise ConversationThreadMemoryError(
                "THREAD_MEMORY_ENVELOPE_MALFORMED"
            ) from exc

    def save_envelope(
        self,
        *,
        partition: ConversationThreadMemoryPartitionV1,
        envelope: ThreadMemoryLifecycleEnvelopeV1,
        expected_revision_id: str | None,
    ) -> bool:
        replacement = self._record(partition=partition, envelope=envelope)
        conditional = self._conditional_store()
        if expected_revision_id is None:
            return conditional.put_if_absent(replacement)

        current = self._store.get(_DOCUMENT_PARTITION, self._row_key(partition))
        if current is None:
            return False
        try:
            current_envelope = _deserialize_envelope(dict(current.data))
        except ConversationThreadMemoryError:
            raise
        if current_envelope.revision_id != expected_revision_id:
            return False
        return conditional.replace_if_match(
            expected=current,
            replacement=replacement,
        )


def _role_to_platform(role: ConversationThreadMemoryMessageRole) -> str:
    return role.value


def _role_from_platform(role: str) -> ConversationThreadMemoryMessageRole:
    return ConversationThreadMemoryMessageRole(role)


def _parse_created_at(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value)
    except (TypeError, ValueError) as exc:
        raise ConversationThreadMemoryError("THREAD_MEMORY_CREATED_AT_INVALID") from exc
    if parsed.tzinfo is None:
        raise ConversationThreadMemoryError("THREAD_MEMORY_CREATED_AT_INVALID")
    offset = parsed.utcoffset()
    if offset is None or offset != timedelta(0):
        raise ConversationThreadMemoryError("THREAD_MEMORY_CREATED_AT_INVALID")
    return parsed


def _to_platform_message(
    message: ConversationThreadMemoryMessageV1,
    *,
    sequence: int,
) -> SessionHistoryMessage:
    entry_id = f"lkw-thread-{sequence}-{uuid4().hex}"
    chat_message = ChatMessage(
        role=_role_to_platform(message.role),  # type: ignore[arg-type]
        content=message.content,
        entry_id=entry_id,
    )
    return session_history_message_from_chat_message(chat_message, sequence=sequence)


def _from_platform_message(
    message: SessionHistoryMessage,
    *,
    created_at: datetime,
) -> ConversationThreadMemoryMessageV1:
    return ConversationThreadMemoryMessageV1(
        role=_role_from_platform(message.role),
        content=message.content,
        created_at=created_at,
    )


def _validate_partition_against_context(
    *,
    partition: ConversationThreadMemoryPartitionV1,
    context: ConversationExecutionContextV1,
) -> None:
    if partition.tenant_id != context.tenant_id:
        raise ConversationThreadMemoryError("THREAD_MEMORY_TENANT_MISMATCH")
    if partition.conversation_context_binding_id != context.conversation_context_binding_id:
        raise ConversationThreadMemoryError("THREAD_MEMORY_BINDING_MISMATCH")
    if partition.canonical_thread_ref != context.canonical_thread_ref:
        raise ConversationThreadMemoryError("THREAD_MEMORY_THREAD_MISMATCH")
    if partition.audience_mode != context.audience_mode:
        raise ConversationThreadMemoryError("THREAD_MEMORY_AUDIENCE_MISMATCH")
    if partition.workspace_id != context.workspace_id:
        raise ConversationThreadMemoryError("THREAD_MEMORY_WORKSPACE_MISMATCH")


def _validate_snapshot_partition(
    *,
    memory_snapshot: ConversationThreadMemorySnapshotV1,
    partition: ConversationThreadMemoryPartitionV1,
) -> None:
    if memory_snapshot.schema_version != _PARTITION_SCHEMA_VERSION:
        raise ConversationThreadMemoryError("THREAD_MEMORY_PARTITION_SCHEMA_MISMATCH")
    if memory_snapshot.tenant_id != partition.tenant_id:
        raise ConversationThreadMemoryError("THREAD_MEMORY_TENANT_MISMATCH")
    if memory_snapshot.conversation_context_binding_id != partition.conversation_context_binding_id:
        raise ConversationThreadMemoryError("THREAD_MEMORY_BINDING_MISMATCH")
    if memory_snapshot.canonical_thread_ref != partition.canonical_thread_ref:
        raise ConversationThreadMemoryError("THREAD_MEMORY_THREAD_MISMATCH")
    if memory_snapshot.audience_mode != partition.audience_mode:
        raise ConversationThreadMemoryError("THREAD_MEMORY_AUDIENCE_MISMATCH")
    if memory_snapshot.workspace_id != partition.workspace_id:
        raise ConversationThreadMemoryError("THREAD_MEMORY_WORKSPACE_MISMATCH")
    if memory_snapshot.snapshot.tenant_id != partition.tenant_id:
        raise ConversationThreadMemoryError("THREAD_MEMORY_SNAPSHOT_IDENTITY_MISMATCH")
    expected_scope = derive_conversation_thread_session_key(
        tenant_id=partition.tenant_id,
        conversation_context_binding_id=partition.conversation_context_binding_id,
        canonical_thread_ref=partition.canonical_thread_ref,
    )
    if memory_snapshot.snapshot.context_scope_id != expected_scope:
        raise ConversationThreadMemoryError("THREAD_MEMORY_SNAPSHOT_IDENTITY_MISMATCH")


def _validate_snapshot_timestamps(memory_snapshot: ConversationThreadMemorySnapshotV1) -> dict[str, datetime]:
    message_ids = {message.message_id for message in memory_snapshot.snapshot.messages}
    created_at_by_id: dict[str, datetime] = {}
    for message_id, raw_value in memory_snapshot.message_created_at:
        if message_id in created_at_by_id:
            raise ConversationThreadMemoryError("THREAD_MEMORY_CREATED_AT_DUPLICATE")
        created_at_by_id[message_id] = _parse_created_at(raw_value)
    for message_id in message_ids:
        if message_id not in created_at_by_id:
            raise ConversationThreadMemoryError("THREAD_MEMORY_CREATED_AT_MISSING")
    for message_id in created_at_by_id:
        if message_id not in message_ids:
            raise ConversationThreadMemoryError("THREAD_MEMORY_CREATED_AT_UNKNOWN")
    return created_at_by_id


def _derive_revision_id(
    *,
    tenant_id: str,
    context_scope_id: str,
    messages: tuple[SessionHistoryMessage, ...],
) -> str:
    payload = {
        "version": 1,
        "tenant_id": tenant_id,
        "context_scope_id": context_scope_id,
        "messages": [
            {
                "message_id": message.message_id,
                "sequence": message.sequence,
                "content_hash": message.content_hash,
            }
            for message in messages
        ],
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"{_REVISION_ID_PREFIX}{digest}"


def _message_byte_length(message: ConversationThreadMemoryMessageV1) -> int:
    return len(message.content.encode("utf-8"))


def _apply_bounds(
    messages: tuple[ConversationThreadMemoryMessageV1, ...],
    *,
    limits: ConversationThreadMemoryLimitsV1,
    now: datetime,
) -> tuple[ConversationThreadMemoryMessageV1, ...]:
    if not messages:
        return ()

    cutoff = now - timedelta(seconds=limits.max_age_seconds)
    age_filtered = tuple(message for message in messages if message.created_at >= cutoff)
    if not age_filtered:
        return ()

    selected: list[ConversationThreadMemoryMessageV1] = []
    total_bytes = 0
    for message in reversed(age_filtered):
        message_bytes = _message_byte_length(message)
        candidate_count = len(selected) + 1
        candidate_bytes = total_bytes + message_bytes
        if candidate_count > limits.max_messages:
            break
        if candidate_bytes > limits.max_bytes:
            break
        selected.append(message)
        total_bytes = candidate_bytes

    if not selected:
        return ()

    newest = age_filtered[-1]
    if _message_byte_length(newest) > limits.max_bytes:
        return ()

    return tuple(reversed(selected))


def _build_snapshot_envelope(
    *,
    partition: ConversationThreadMemoryPartitionV1,
    context_scope_id: str,
    messages: tuple[SessionHistoryMessage, ...],
    created_at_entries: tuple[tuple[str, str], ...],
) -> ConversationThreadMemorySnapshotV1:
    revision_id = _derive_revision_id(
        tenant_id=partition.tenant_id,
        context_scope_id=context_scope_id,
        messages=messages,
    )
    snapshot = SessionHistorySnapshot(
        tenant_id=partition.tenant_id,
        context_scope_id=context_scope_id,
        revision_id=revision_id,
        messages=messages,
    )
    return ConversationThreadMemorySnapshotV1(
        schema_version=_PARTITION_SCHEMA_VERSION,
        tenant_id=partition.tenant_id,
        conversation_context_binding_id=partition.conversation_context_binding_id,
        canonical_thread_ref=partition.canonical_thread_ref,
        audience_mode=partition.audience_mode,
        workspace_id=partition.workspace_id,
        snapshot=snapshot,
        message_created_at=created_at_entries,
    )


def _next_message_sequence(messages: tuple[SessionHistoryMessage, ...]) -> int:
    if not messages:
        return 0
    return messages[-1].sequence + 1


def _build_appended_snapshot(
    *,
    partition: ConversationThreadMemoryPartitionV1,
    memory_snapshot: ConversationThreadMemorySnapshotV1 | None,
    messages: tuple[ConversationThreadMemoryMessageV1, ...],
) -> ConversationThreadMemorySnapshotV1:
    context_scope_id = derive_conversation_thread_session_key(
        tenant_id=partition.tenant_id,
        conversation_context_binding_id=partition.conversation_context_binding_id,
        canonical_thread_ref=partition.canonical_thread_ref,
    )
    if memory_snapshot is not None:
        _validate_snapshot_partition(memory_snapshot=memory_snapshot, partition=partition)
        _validate_snapshot_timestamps(memory_snapshot)
        existing_messages = memory_snapshot.snapshot.messages
        created_at_entries = list(memory_snapshot.message_created_at)
    else:
        existing_messages = ()
        created_at_entries = []

    next_sequence = _next_message_sequence(existing_messages)
    platform_messages: list[SessionHistoryMessage] = []
    for offset, message in enumerate(messages):
        platform_message = _to_platform_message(message, sequence=next_sequence + offset)
        platform_messages.append(platform_message)
        created_at_entries.append((platform_message.message_id, message.created_at.isoformat()))

    return _build_snapshot_envelope(
        partition=partition,
        context_scope_id=context_scope_id,
        messages=existing_messages + tuple(platform_messages),
        created_at_entries=tuple(created_at_entries),
    )


class SessionHistorySnapshotConversationThreadMemoryAdapter:
    """Bounded thread memory adapter with conflict-safe lifecycle-port appends."""

    def __init__(self, *, port: ThreadMemoryLifecyclePort) -> None:
        self._port = port

    def load_bounded_history_from_port(
        self,
        *,
        context: ConversationExecutionContextV1,
        limits: ConversationThreadMemoryLimitsV1,
        now: datetime,
    ) -> tuple[ConversationThreadMemoryMessageV1, ...]:
        partition = ConversationThreadMemoryPartitionV1.from_execution_context(context)
        loaded = self._port.load_envelope(partition=partition)
        return self.load_bounded_history(
            context=context,
            memory_snapshot=loaded.memory_snapshot if loaded is not None else None,
            limits=limits,
            now=now,
        )

    @staticmethod
    def load_bounded_history(
        *,
        context: ConversationExecutionContextV1,
        memory_snapshot: ConversationThreadMemorySnapshotV1 | None,
        limits: ConversationThreadMemoryLimitsV1,
        now: datetime,
    ) -> tuple[ConversationThreadMemoryMessageV1, ...]:
        if memory_snapshot is None:
            return ()
        partition = ConversationThreadMemoryPartitionV1.from_execution_context(context)
        _validate_snapshot_partition(memory_snapshot=memory_snapshot, partition=partition)
        created_at_by_id = _validate_snapshot_timestamps(memory_snapshot)
        converted: list[ConversationThreadMemoryMessageV1] = []
        for platform_message in memory_snapshot.snapshot.messages:
            converted.append(
                _from_platform_message(
                    platform_message,
                    created_at=created_at_by_id[platform_message.message_id],
                )
            )
        return _apply_bounds(tuple(converted), limits=limits, now=now)

    def _append_messages(
        self,
        *,
        context: ConversationExecutionContextV1,
        messages: tuple[ConversationThreadMemoryMessageV1, ...],
        exchange_id: str | None = None,
    ) -> ConversationThreadMemorySnapshotV1:
        partition = ConversationThreadMemoryPartitionV1.from_execution_context(context)
        _validate_partition_against_context(partition=partition, context=context)
        loaded = self._port.load_envelope(partition=partition)
        if loaded is not None and exchange_id in loaded.applied_exchange_ids:
            return loaded.memory_snapshot
        memory_snapshot = loaded.memory_snapshot if loaded is not None else None
        expected_revision_id = loaded.revision_id if loaded is not None else None
        new_snapshot = _build_appended_snapshot(
            partition=partition,
            memory_snapshot=memory_snapshot,
            messages=messages,
        )
        new_envelope = ThreadMemoryLifecycleEnvelopeV1(memory_snapshot=new_snapshot)
        applied_exchange_ids = (
            loaded.applied_exchange_ids if loaded is not None else ()
        )
        if exchange_id is not None:
            applied_exchange_ids = (
                *applied_exchange_ids,
                exchange_id,
            )[-_MAX_APPLIED_EXCHANGE_IDS:]
            new_envelope = ThreadMemoryLifecycleEnvelopeV1(
                memory_snapshot=new_snapshot,
                applied_exchange_ids=applied_exchange_ids,
            )
        if not self._port.save_envelope(
            partition=partition,
            envelope=new_envelope,
            expected_revision_id=expected_revision_id,
        ):
            raise ConversationThreadMemoryError("THREAD_MEMORY_REVISION_CONFLICT")
        return new_snapshot

    def append_message(
        self,
        *,
        context: ConversationExecutionContextV1,
        message: ConversationThreadMemoryMessageV1,
    ) -> ConversationThreadMemorySnapshotV1:
        return self._append_messages(context=context, messages=(message,))

    def append_exchange(
        self,
        *,
        context: ConversationExecutionContextV1,
        user_message: ConversationThreadMemoryMessageV1,
        assistant_message: ConversationThreadMemoryMessageV1,
        exchange_id: str | None = None,
    ) -> ConversationThreadMemorySnapshotV1:
        if user_message.role is not ConversationThreadMemoryMessageRole.USER:
            raise ConversationThreadMemoryError("THREAD_MEMORY_EXCHANGE_USER_ROLE_REQUIRED")
        if assistant_message.role is not ConversationThreadMemoryMessageRole.ASSISTANT:
            raise ConversationThreadMemoryError("THREAD_MEMORY_EXCHANGE_ASSISTANT_ROLE_REQUIRED")
        return self._append_messages(
            context=context,
            messages=(user_message, assistant_message),
            exchange_id=exchange_id,
        )
