# © Artur Czarnecki. All rights reserved.

"""Bounded Conversation thread memory adapter over SessionHistorySnapshot (LKW-CONVERSATION-CONTEXT-1B1)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from uuid import uuid4

from intergrax.context.session_history import (
    SessionHistoryMessage,
    SessionHistorySnapshot,
    session_history_message_from_chat_message,
)
from intergrax.llm.messages import ChatMessage
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


def _role_to_platform(role: ConversationThreadMemoryMessageRole) -> str:
    return role.value


def _role_from_platform(role: str) -> ConversationThreadMemoryMessageRole:
    return ConversationThreadMemoryMessageRole(role)


def _parse_created_at(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
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


class SessionHistorySnapshotConversationThreadMemoryAdapter:
    """Pure immutable adapter over SessionHistorySnapshot for bounded thread memory."""

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

    @staticmethod
    def append_message(
        *,
        context: ConversationExecutionContextV1,
        memory_snapshot: ConversationThreadMemorySnapshotV1 | None,
        message: ConversationThreadMemoryMessageV1,
    ) -> ConversationThreadMemorySnapshotV1:
        partition = ConversationThreadMemoryPartitionV1.from_execution_context(context)
        _validate_partition_against_context(partition=partition, context=context)
        context_scope_id = derive_conversation_thread_session_key(
            tenant_id=partition.tenant_id,
            conversation_context_binding_id=partition.conversation_context_binding_id,
            canonical_thread_ref=partition.canonical_thread_ref,
        )
        if memory_snapshot is not None:
            _validate_snapshot_partition(memory_snapshot=memory_snapshot, partition=partition)
            _validate_snapshot_timestamps(memory_snapshot)
            next_sequence = memory_snapshot.snapshot.messages[-1].sequence + 1
            existing_messages = memory_snapshot.snapshot.messages
            created_at_entries = list(memory_snapshot.message_created_at)
        else:
            next_sequence = 0
            existing_messages = ()
            created_at_entries = []

        platform_message = _to_platform_message(message, sequence=next_sequence)
        created_at_entries.append((platform_message.message_id, message.created_at.isoformat()))
        messages = existing_messages + (platform_message,)
        return _build_snapshot_envelope(
            partition=partition,
            context_scope_id=context_scope_id,
            messages=messages,
            created_at_entries=tuple(created_at_entries),
        )

    @staticmethod
    def append_exchange(
        *,
        context: ConversationExecutionContextV1,
        memory_snapshot: ConversationThreadMemorySnapshotV1 | None,
        user_message: ConversationThreadMemoryMessageV1,
        assistant_message: ConversationThreadMemoryMessageV1,
    ) -> ConversationThreadMemorySnapshotV1:
        if user_message.role is not ConversationThreadMemoryMessageRole.USER:
            raise ConversationThreadMemoryError("THREAD_MEMORY_EXCHANGE_USER_ROLE_REQUIRED")
        if assistant_message.role is not ConversationThreadMemoryMessageRole.ASSISTANT:
            raise ConversationThreadMemoryError("THREAD_MEMORY_EXCHANGE_ASSISTANT_ROLE_REQUIRED")

        partition = ConversationThreadMemoryPartitionV1.from_execution_context(context)
        _validate_partition_against_context(partition=partition, context=context)
        context_scope_id = derive_conversation_thread_session_key(
            tenant_id=partition.tenant_id,
            conversation_context_binding_id=partition.conversation_context_binding_id,
            canonical_thread_ref=partition.canonical_thread_ref,
        )
        if memory_snapshot is not None:
            _validate_snapshot_partition(memory_snapshot=memory_snapshot, partition=partition)
            _validate_snapshot_timestamps(memory_snapshot)
            next_sequence = memory_snapshot.snapshot.messages[-1].sequence + 1
            existing_messages = memory_snapshot.snapshot.messages
            created_at_entries = list(memory_snapshot.message_created_at)
        else:
            next_sequence = 0
            existing_messages = ()
            created_at_entries = []

        user_platform = _to_platform_message(user_message, sequence=next_sequence)
        assistant_platform = _to_platform_message(
            assistant_message,
            sequence=next_sequence + 1,
        )
        created_at_entries.append((user_platform.message_id, user_message.created_at.isoformat()))
        created_at_entries.append(
            (assistant_platform.message_id, assistant_message.created_at.isoformat())
        )
        messages = existing_messages + (user_platform, assistant_platform)
        return _build_snapshot_envelope(
            partition=partition,
            context_scope_id=context_scope_id,
            messages=messages,
            created_at_entries=tuple(created_at_entries),
        )
