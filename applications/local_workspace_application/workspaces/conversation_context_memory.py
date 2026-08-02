# © Artur Czarnecki. All rights reserved.

"""Bounded Conversation thread memory adapter over Context Lifecycle (LKW-CONVERSATION-CONTEXT-1B1)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Protocol, runtime_checkable
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
class ThreadMemoryLifecycleEnvelopeV1:
    """Adapter-owned partition metadata envelope carried with a platform snapshot."""

    schema_version: int
    tenant_id: str
    conversation_context_binding_id: str
    canonical_thread_ref: str
    audience_mode: ConversationAudienceMode
    workspace_id: str
    snapshot: SessionHistorySnapshot
    message_created_at: tuple[tuple[str, str], ...]


@runtime_checkable
class ContextLifecycleThreadMemoryPort(Protocol):
    """Minimal public Context Lifecycle surface for bounded thread-memory persistence."""

    def load_envelope(
        self,
        *,
        tenant_id: str,
        context_scope_id: str,
    ) -> ThreadMemoryLifecycleEnvelopeV1 | None:
        """Load the stored thread-memory envelope for a context scope."""

    def save_envelope(
        self,
        *,
        envelope: ThreadMemoryLifecycleEnvelopeV1,
    ) -> None:
        """Persist the thread-memory envelope for a context scope."""


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


def _to_lifecycle_message(
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


def _from_lifecycle_message(
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


def _validate_envelope_partition(
    *,
    envelope: ThreadMemoryLifecycleEnvelopeV1,
    partition: ConversationThreadMemoryPartitionV1,
) -> None:
    if envelope.schema_version != _PARTITION_SCHEMA_VERSION:
        raise ConversationThreadMemoryError("THREAD_MEMORY_PARTITION_SCHEMA_MISMATCH")
    if envelope.tenant_id != partition.tenant_id:
        raise ConversationThreadMemoryError("THREAD_MEMORY_TENANT_MISMATCH")
    if envelope.conversation_context_binding_id != partition.conversation_context_binding_id:
        raise ConversationThreadMemoryError("THREAD_MEMORY_BINDING_MISMATCH")
    if envelope.canonical_thread_ref != partition.canonical_thread_ref:
        raise ConversationThreadMemoryError("THREAD_MEMORY_THREAD_MISMATCH")
    if envelope.audience_mode != partition.audience_mode:
        raise ConversationThreadMemoryError("THREAD_MEMORY_AUDIENCE_MISMATCH")
    if envelope.workspace_id != partition.workspace_id:
        raise ConversationThreadMemoryError("THREAD_MEMORY_WORKSPACE_MISMATCH")
    if envelope.snapshot.tenant_id != partition.tenant_id:
        raise ConversationThreadMemoryError("THREAD_MEMORY_TENANT_MISMATCH")
    expected_scope = derive_conversation_thread_session_key(
        tenant_id=partition.tenant_id,
        conversation_context_binding_id=partition.conversation_context_binding_id,
        canonical_thread_ref=partition.canonical_thread_ref,
    )
    if envelope.snapshot.context_scope_id != expected_scope:
        raise ConversationThreadMemoryError("THREAD_MEMORY_THREAD_MISMATCH")


def _created_at_map(envelope: ThreadMemoryLifecycleEnvelopeV1) -> dict[str, datetime]:
    return {message_id: _parse_created_at(value) for message_id, value in envelope.message_created_at}


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


class ContextLifecycleConversationThreadMemoryAdapter:
    """Provider-neutral bounded thread-memory adapter over Context Lifecycle snapshots."""

    def __init__(self, lifecycle_port: ContextLifecycleThreadMemoryPort) -> None:
        self._lifecycle_port = lifecycle_port

    def load_bounded_history(
        self,
        *,
        context: ConversationExecutionContextV1,
        limits: ConversationThreadMemoryLimitsV1,
        now: datetime,
    ) -> tuple[ConversationThreadMemoryMessageV1, ...]:
        partition = ConversationThreadMemoryPartitionV1.from_execution_context(context)
        context_scope_id = derive_conversation_thread_session_key(
            tenant_id=partition.tenant_id,
            conversation_context_binding_id=partition.conversation_context_binding_id,
            canonical_thread_ref=partition.canonical_thread_ref,
        )
        envelope = self._lifecycle_port.load_envelope(
            tenant_id=partition.tenant_id,
            context_scope_id=context_scope_id,
        )
        if envelope is None:
            return ()
        _validate_envelope_partition(envelope=envelope, partition=partition)
        created_at_by_id = _created_at_map(envelope)
        converted: list[ConversationThreadMemoryMessageV1] = []
        for platform_message in envelope.snapshot.messages:
            created_at = created_at_by_id.get(platform_message.message_id)
            if created_at is None:
                raise ConversationThreadMemoryError("THREAD_MEMORY_CREATED_AT_MISSING")
            converted.append(_from_lifecycle_message(platform_message, created_at=created_at))
        bounded = _apply_bounds(tuple(converted), limits=limits, now=now)
        return bounded

    def append_message(
        self,
        *,
        context: ConversationExecutionContextV1,
        message: ConversationThreadMemoryMessageV1,
    ) -> None:
        partition = ConversationThreadMemoryPartitionV1.from_execution_context(context)
        _validate_partition_against_context(partition=partition, context=context)
        context_scope_id = derive_conversation_thread_session_key(
            tenant_id=partition.tenant_id,
            conversation_context_binding_id=partition.conversation_context_binding_id,
            canonical_thread_ref=partition.canonical_thread_ref,
        )
        existing = self._lifecycle_port.load_envelope(
            tenant_id=partition.tenant_id,
            context_scope_id=context_scope_id,
        )
        if existing is not None:
            _validate_envelope_partition(envelope=existing, partition=partition)
            next_sequence = len(existing.snapshot.messages)
            revision_id = f"rev-{next_sequence + 1}"
            created_at_entries = list(existing.message_created_at)
        else:
            next_sequence = 0
            revision_id = "rev-1"
            created_at_entries = []

        platform_message = _to_lifecycle_message(message, sequence=next_sequence)
        created_at_entries.append((platform_message.message_id, message.created_at.isoformat()))
        messages = tuple(existing.snapshot.messages) if existing is not None else ()
        snapshot = SessionHistorySnapshot(
            tenant_id=partition.tenant_id,
            context_scope_id=context_scope_id,
            revision_id=revision_id,
            messages=messages + (platform_message,),
        )
        envelope = ThreadMemoryLifecycleEnvelopeV1(
            schema_version=_PARTITION_SCHEMA_VERSION,
            tenant_id=partition.tenant_id,
            conversation_context_binding_id=partition.conversation_context_binding_id,
            canonical_thread_ref=partition.canonical_thread_ref,
            audience_mode=partition.audience_mode,
            workspace_id=partition.workspace_id,
            snapshot=snapshot,
            message_created_at=tuple(created_at_entries),
        )
        self._lifecycle_port.save_envelope(envelope=envelope)

    def append_exchange(
        self,
        *,
        context: ConversationExecutionContextV1,
        user_message: ConversationThreadMemoryMessageV1,
        assistant_message: ConversationThreadMemoryMessageV1,
    ) -> None:
        if user_message.role is not ConversationThreadMemoryMessageRole.USER:
            raise ConversationThreadMemoryError("THREAD_MEMORY_EXCHANGE_USER_ROLE_REQUIRED")
        if assistant_message.role is not ConversationThreadMemoryMessageRole.ASSISTANT:
            raise ConversationThreadMemoryError("THREAD_MEMORY_EXCHANGE_ASSISTANT_ROLE_REQUIRED")
        self.append_message(context=context, message=user_message)
        self.append_message(context=context, message=assistant_message)
