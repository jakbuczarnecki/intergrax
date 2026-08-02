# © Artur Czarnecki. All rights reserved.

"""Unit tests for Conversation thread memory snapshot adapter."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta, timezone

import pytest

from intergrax.context.session_history import SessionHistoryMessage, SessionHistorySnapshot
from local_workspace_application.workspaces.conversation_context_execution import (
    build_conversation_execution_context,
)
from local_workspace_application.workspaces.conversation_context_memory import (
    ConversationThreadMemoryError,
    ConversationThreadMemoryPartitionV1,
    ConversationThreadMemorySnapshotV1,
    InMemoryThreadMemoryLifecyclePort,
    SessionHistorySnapshotConversationThreadMemoryAdapter,
    ThreadMemoryLifecycleEnvelopeV1,
    ThreadMemoryLifecyclePort,
    _build_snapshot_envelope,
    derive_conversation_thread_session_key,
)
from local_workspace_application.workspaces.conversation_context_models import (
    ConversationActivationPolicy,
    ConversationAudienceMode,
    ConversationProductCapability,
    ConversationThreadContextPolicy,
    ConversationThreadMemoryLimitsV1,
    ConversationThreadMemoryMessageRole,
    ConversationThreadMemoryMessageV1,
    ResolvedConversationWorkspaceContextV1,
)

pytestmark = pytest.mark.unit

_NOW = datetime(2024, 6, 1, 12, 0, tzinfo=UTC)
_TENANT = "tenant-a"
_BINDING = "binding-1"
_BINDING_B = "binding-2"
_WORKSPACE = "workspace-1"
_WORKSPACE_B = "workspace-2"
_PRINCIPAL = "principal.alice"
_THREAD = "thread-1"
_THREAD_B = "thread-2"


def _resolved(
    *,
    audience: ConversationAudienceMode = ConversationAudienceMode.PERSONAL,
    binding: str = _BINDING,
    workspace: str = _WORKSPACE,
    thread: str = _THREAD,
    tenant: str = _TENANT,
) -> ResolvedConversationWorkspaceContextV1:
    return ResolvedConversationWorkspaceContextV1(
        tenant_id=tenant,
        conversation_context_binding_id=binding,
        audience_mode=audience,
        workspace_id=workspace,
        principal_ref=_PRINCIPAL,
        canonical_thread_ref=thread,
        activation_policy=ConversationActivationPolicy.ALWAYS,
        thread_context_policy=ConversationThreadContextPolicy.CURRENT_THREAD_BOUNDED,
    )


def _context(
    *,
    audience: ConversationAudienceMode = ConversationAudienceMode.PERSONAL,
    binding: str = _BINDING,
    workspace: str = _WORKSPACE,
    thread: str = _THREAD,
    tenant: str = _TENANT,
) -> object:
    resolved = _resolved(
        audience=audience,
        binding=binding,
        workspace=workspace,
        thread=thread,
        tenant=tenant,
    )
    if audience is ConversationAudienceMode.SHARED:
        return build_conversation_execution_context(resolved=resolved)
    return build_conversation_execution_context(
        resolved=resolved,
        personal_allowed_capabilities=frozenset({ConversationProductCapability.READ_ONLY_ASK}),
    )


def _message(
    *,
    role: ConversationThreadMemoryMessageRole = ConversationThreadMemoryMessageRole.USER,
    content: str = "hello",
    created_at: datetime = _NOW,
) -> ConversationThreadMemoryMessageV1:
    return ConversationThreadMemoryMessageV1(
        role=role,
        content=content,
        created_at=created_at,
    )


def _limits(**overrides: int) -> ConversationThreadMemoryLimitsV1:
    payload = {"max_messages": 50, "max_bytes": 10_000, "max_age_seconds": 86_400}
    payload.update(overrides)
    return ConversationThreadMemoryLimitsV1(**payload)  # type: ignore[arg-type]


def _scope(
    *,
    tenant: str = _TENANT,
    binding: str = _BINDING,
    thread: str = _THREAD,
) -> str:
    return derive_conversation_thread_session_key(
        tenant_id=tenant,
        conversation_context_binding_id=binding,
        canonical_thread_ref=thread,
    )


def _adapter(
    port: InMemoryThreadMemoryLifecyclePort | None = None,
) -> SessionHistorySnapshotConversationThreadMemoryAdapter:
    return SessionHistorySnapshotConversationThreadMemoryAdapter(
        port=port or InMemoryThreadMemoryLifecyclePort(),
    )


def _tamper_envelope(
    envelope: ConversationThreadMemorySnapshotV1,
    **overrides: object,
) -> ConversationThreadMemorySnapshotV1:
    fields = {
        "schema_version": envelope.schema_version,
        "tenant_id": envelope.tenant_id,
        "conversation_context_binding_id": envelope.conversation_context_binding_id,
        "canonical_thread_ref": envelope.canonical_thread_ref,
        "audience_mode": envelope.audience_mode,
        "workspace_id": envelope.workspace_id,
        "snapshot": envelope.snapshot,
        "message_created_at": envelope.message_created_at,
    }
    fields.update(overrides)
    return ConversationThreadMemorySnapshotV1(**fields)  # type: ignore[arg-type]



def _platform_message(
    *,
    sequence: int,
    role: str,
    content: str,
    message_id: str,
) -> SessionHistoryMessage:
    return SessionHistoryMessage(
        message_id=message_id,
        sequence=sequence,
        role=role,  # type: ignore[arg-type]
        content=content,
    )


def _seed_non_contiguous_snapshot(
    context: object,
) -> tuple[InMemoryThreadMemoryLifecyclePort, ConversationThreadMemorySnapshotV1]:
    partition = ConversationThreadMemoryPartitionV1.from_execution_context(context)  # type: ignore[arg-type]
    messages = (
        _platform_message(sequence=0, role="user", content="m0", message_id="msg-0"),
        _platform_message(sequence=3, role="assistant", content="m3", message_id="msg-3"),
        _platform_message(sequence=8, role="user", content="m8", message_id="msg-8"),
    )
    created_at_entries = (
        ("msg-0", _NOW.isoformat()),
        ("msg-3", (_NOW + timedelta(seconds=1)).isoformat()),
        ("msg-8", (_NOW + timedelta(seconds=2)).isoformat()),
    )
    snapshot = _build_snapshot_envelope(
        partition=partition,
        context_scope_id=_scope(),
        messages=messages,
        created_at_entries=created_at_entries,
    )
    port = InMemoryThreadMemoryLifecyclePort()
    port.save_envelope(
        partition=partition,
        envelope=ThreadMemoryLifecycleEnvelopeV1(memory_snapshot=snapshot),
        expected_revision_id=None,
    )
    return port, snapshot


@dataclass
class _RecordingLifecyclePort:
    inner: InMemoryThreadMemoryLifecyclePort
    load_count: int = 0
    save_count: int = 0
    last_expected_revision_id: str | None = None

    def load_envelope(
        self,
        *,
        partition: ConversationThreadMemoryPartitionV1,
    ) -> ThreadMemoryLifecycleEnvelopeV1 | None:
        self.load_count += 1
        return self.inner.load_envelope(partition=partition)

    def save_envelope(
        self,
        *,
        partition: ConversationThreadMemoryPartitionV1,
        envelope: ThreadMemoryLifecycleEnvelopeV1,
        expected_revision_id: str | None,
    ) -> bool:
        self.save_count += 1
        self.last_expected_revision_id = expected_revision_id
        return self.inner.save_envelope(
            partition=partition,
            envelope=envelope,
            expected_revision_id=expected_revision_id,
        )


class _ConflictOnSavePort:
    def __init__(self, *, inner: InMemoryThreadMemoryLifecyclePort) -> None:
        self._inner = inner

    def load_envelope(
        self,
        *,
        partition: ConversationThreadMemoryPartitionV1,
    ) -> ThreadMemoryLifecycleEnvelopeV1 | None:
        return self._inner.load_envelope(partition=partition)

    def save_envelope(
        self,
        *,
        partition: ConversationThreadMemoryPartitionV1,
        envelope: ThreadMemoryLifecycleEnvelopeV1,
        expected_revision_id: str | None,
    ) -> bool:
        return False


def test_non_contiguous_sequences_load_and_append() -> None:
    context = _context()
    port, seeded = _seed_non_contiguous_snapshot(context)
    adapter = _adapter(port=port)
    history = SessionHistorySnapshotConversationThreadMemoryAdapter.load_bounded_history(
        context=context,  # type: ignore[arg-type]
        memory_snapshot=seeded,
        limits=_limits(),
        now=_NOW + timedelta(seconds=10),
    )
    assert [item.content for item in history] == ["m0", "m3", "m8"]

    original_ids = [message.message_id for message in seeded.snapshot.messages]
    original_sequences = [message.sequence for message in seeded.snapshot.messages]
    updated = adapter.append_message(
        context=context,  # type: ignore[arg-type]
        message=_message(content="next", created_at=_NOW + timedelta(seconds=3)),
    )
    sequences = [message.sequence for message in updated.snapshot.messages]
    message_ids = [message.message_id for message in updated.snapshot.messages]
    assert sequences == [0, 3, 8, 9]
    assert message_ids[:3] == original_ids
    assert updated.snapshot.messages[:3] == seeded.snapshot.messages
    assert original_sequences == [0, 3, 8]


def test_append_message_passes_expected_revision_to_port() -> None:
    context = _context()
    inner = InMemoryThreadMemoryLifecyclePort()
    recording: ThreadMemoryLifecyclePort = _RecordingLifecyclePort(inner=inner)
    adapter = SessionHistorySnapshotConversationThreadMemoryAdapter(port=recording)
    first = adapter.append_message(
        context=context,  # type: ignore[arg-type]
        message=_message(content="first"),
    )
    recording_port = recording
    assert isinstance(recording_port, _RecordingLifecyclePort)
    assert recording_port.last_expected_revision_id is None

    adapter.append_message(
        context=context,  # type: ignore[arg-type]
        message=_message(content="second", created_at=_NOW + timedelta(seconds=1)),
    )
    assert recording_port.last_expected_revision_id == first.snapshot.revision_id


def test_matching_revision_append_succeeds() -> None:
    context = _context()
    adapter = _adapter()
    first = adapter.append_message(
        context=context,  # type: ignore[arg-type]
        message=_message(content="first"),
    )
    second = adapter.append_message(
        context=context,  # type: ignore[arg-type]
        message=_message(content="second", created_at=_NOW + timedelta(seconds=1)),
    )
    assert len(second.snapshot.messages) == 2
    assert second.snapshot.revision_id != first.snapshot.revision_id


def test_mismatching_revision_raises_conflict() -> None:
    context = _context()
    port = InMemoryThreadMemoryLifecyclePort()
    adapter = _adapter(port=port)
    adapter.append_message(
        context=context,  # type: ignore[arg-type]
        message=_message(content="first"),
    )
    partition = ConversationThreadMemoryPartitionV1.from_execution_context(context)  # type: ignore[arg-type]
    stale_revision = "lkw-thread-revision:v1:stale-revision"
    stale_messages = (
        _platform_message(sequence=0, role="user", content="stale", message_id="stale-0"),
    )
    stale_snapshot = _build_snapshot_envelope(
        partition=partition,
        context_scope_id=_scope(),
        messages=stale_messages,
        created_at_entries=(("stale-0", _NOW.isoformat()),),
    )
    stale_envelope = ThreadMemoryLifecycleEnvelopeV1(
        memory_snapshot=_tamper_envelope(
            stale_snapshot,
            snapshot=SessionHistorySnapshot(
                tenant_id=stale_snapshot.snapshot.tenant_id,
                context_scope_id=stale_snapshot.snapshot.context_scope_id,
                revision_id=stale_revision,
                messages=stale_snapshot.snapshot.messages,
            ),
        )
    )

    class _StaleLoadPort:
        def load_envelope(
            self,
            *,
            partition: ConversationThreadMemoryPartitionV1,
        ) -> ThreadMemoryLifecycleEnvelopeV1 | None:
            return stale_envelope

        def save_envelope(
            self,
            *,
            partition: ConversationThreadMemoryPartitionV1,
            envelope: ThreadMemoryLifecycleEnvelopeV1,
            expected_revision_id: str | None,
        ) -> bool:
            return port.save_envelope(
                partition=partition,
                envelope=envelope,
                expected_revision_id=expected_revision_id,
            )

    conflict_adapter = SessionHistorySnapshotConversationThreadMemoryAdapter(port=_StaleLoadPort())
    with pytest.raises(ConversationThreadMemoryError) as exc_info:
        conflict_adapter.append_message(
            context=context,  # type: ignore[arg-type]
            message=_message(content="lost", created_at=_NOW + timedelta(seconds=1)),
        )
    assert exc_info.value.error_code == "THREAD_MEMORY_REVISION_CONFLICT"
    stored = port.load_envelope(partition=partition)
    assert stored is not None
    assert stored.revision_id != stale_revision
    assert len(stored.memory_snapshot.snapshot.messages) == 1
    assert stored.memory_snapshot.snapshot.messages[0].content == "first"


def test_append_exchange_performs_one_load_and_one_save() -> None:
    context = _context()
    inner = InMemoryThreadMemoryLifecyclePort()
    recording: ThreadMemoryLifecyclePort = _RecordingLifecyclePort(inner=inner)
    adapter = SessionHistorySnapshotConversationThreadMemoryAdapter(port=recording)
    adapter.append_exchange(
        context=context,  # type: ignore[arg-type]
        user_message=_message(role=ConversationThreadMemoryMessageRole.USER, content="q"),
        assistant_message=_message(
            role=ConversationThreadMemoryMessageRole.ASSISTANT,
            content="a",
            created_at=_NOW + timedelta(seconds=1),
        ),
    )
    recording_port = recording
    assert isinstance(recording_port, _RecordingLifecyclePort)
    assert recording_port.load_count == 1
    assert recording_port.save_count == 1


def test_append_exchange_conflict_persists_neither_half() -> None:
    context = _context()
    inner = InMemoryThreadMemoryLifecyclePort()
    adapter = SessionHistorySnapshotConversationThreadMemoryAdapter(
        port=_ConflictOnSavePort(inner=inner),
    )
    with pytest.raises(ConversationThreadMemoryError) as exc_info:
        adapter.append_exchange(
            context=context,  # type: ignore[arg-type]
            user_message=_message(role=ConversationThreadMemoryMessageRole.USER, content="q"),
            assistant_message=_message(
                role=ConversationThreadMemoryMessageRole.ASSISTANT,
                content="a",
                created_at=_NOW + timedelta(seconds=1),
            ),
        )
    assert exc_info.value.error_code == "THREAD_MEMORY_REVISION_CONFLICT"
    partition = ConversationThreadMemoryPartitionV1.from_execution_context(context)  # type: ignore[arg-type]
    assert inner.load_envelope(partition=partition) is None

def test_none_snapshot_loads_as_empty_history() -> None:
    context = _context()
    history = SessionHistorySnapshotConversationThreadMemoryAdapter.load_bounded_history(
        context=context,  # type: ignore[arg-type]
        memory_snapshot=None,
        limits=_limits(),
        now=_NOW,
    )
    assert history == ()


def test_append_to_empty_returns_new_snapshot() -> None:
    context = _context()
    snapshot = _adapter().append_message(
        context=context,  # type: ignore[arg-type]
        message=_message(content="first"),
    )
    assert snapshot is not None
    assert len(snapshot.snapshot.messages) == 1
    assert snapshot.snapshot.revision_id.startswith("lkw-thread-revision:v1:")


def test_original_snapshot_unchanged_after_append() -> None:
    context = _context()
    adapter = _adapter()
    first = adapter.append_message(
        context=context,  # type: ignore[arg-type]
        message=_message(content="first"),
    )
    original_revision = first.snapshot.revision_id
    original_message_count = len(first.snapshot.messages)
    adapter.append_message(
        context=context,  # type: ignore[arg-type]
        message=_message(content="second", created_at=_NOW + timedelta(seconds=1)),
    )
    assert first.snapshot.revision_id == original_revision
    assert len(first.snapshot.messages) == original_message_count


def test_append_uses_last_sequence_plus_one() -> None:
    context = _context()
    adapter = _adapter()
    snapshot = adapter.append_message(
        context=context,  # type: ignore[arg-type]
        message=_message(content="one"),
    )
    snapshot = adapter.append_message(
        context=context,  # type: ignore[arg-type]
        message=_message(content="two", created_at=_NOW + timedelta(seconds=1)),
    )
    sequences = [message.sequence for message in snapshot.snapshot.messages]
    assert sequences == [0, 1]


def test_append_exchange_creates_one_result_with_user_then_assistant() -> None:
    context = _context()
    snapshot = _adapter().append_exchange(
        context=context,  # type: ignore[arg-type]
        user_message=_message(
            role=ConversationThreadMemoryMessageRole.USER,
            content="question",
        ),
        assistant_message=_message(
            role=ConversationThreadMemoryMessageRole.ASSISTANT,
            content="answer",
            created_at=_NOW + timedelta(seconds=1),
        ),
    )
    assert len(snapshot.snapshot.messages) == 2
    assert snapshot.snapshot.messages[0].role == "user"
    assert snapshot.snapshot.messages[1].role == "assistant"
    assert snapshot.snapshot.messages[0].sequence == 0
    assert snapshot.snapshot.messages[1].sequence == 1


def test_invalid_assistant_role_fails_before_result() -> None:
    context = _context()
    port = InMemoryThreadMemoryLifecyclePort()
    adapter = _adapter(port=port)
    with pytest.raises(ConversationThreadMemoryError) as exc_info:
        adapter.append_exchange(
            context=context,  # type: ignore[arg-type]
                user_message=_message(role=ConversationThreadMemoryMessageRole.USER, content="q"),
            assistant_message=_message(role=ConversationThreadMemoryMessageRole.USER, content="bad"),
        )
    assert exc_info.value.error_code == "THREAD_MEMORY_EXCHANGE_ASSISTANT_ROLE_REQUIRED"
    partition = ConversationThreadMemoryPartitionV1.from_execution_context(context)  # type: ignore[arg-type]
    assert port.load_envelope(partition=partition) is None


def test_invalid_user_role_fails_before_result() -> None:
    context = _context()
    port = InMemoryThreadMemoryLifecyclePort()
    adapter = _adapter(port=port)
    with pytest.raises(ConversationThreadMemoryError) as exc_info:
        adapter.append_exchange(
            context=context,  # type: ignore[arg-type]
                user_message=_message(role=ConversationThreadMemoryMessageRole.ASSISTANT, content="bad"),
            assistant_message=_message(
                role=ConversationThreadMemoryMessageRole.ASSISTANT,
                content="answer",
            ),
        )
    assert exc_info.value.error_code == "THREAD_MEMORY_EXCHANGE_USER_ROLE_REQUIRED"
    partition = ConversationThreadMemoryPartitionV1.from_execution_context(context)  # type: ignore[arg-type]
    assert port.load_envelope(partition=partition) is None


def test_same_resulting_history_gives_same_revision_id() -> None:
    context = _context()
    snapshot = _adapter().append_message(
        context=context,  # type: ignore[arg-type]
        message=_message(content="stable"),
    )
    partition = ConversationThreadMemoryPartitionV1.from_execution_context(context)  # type: ignore[arg-type]
    rebuilt = _build_snapshot_envelope(
        partition=partition,
        context_scope_id=_scope(),
        messages=snapshot.snapshot.messages,
        created_at_entries=snapshot.message_created_at,
    )
    assert snapshot.snapshot.revision_id == rebuilt.snapshot.revision_id


def test_different_content_gives_different_revision_id() -> None:
    context = _context()
    adapter = _adapter()
    first = adapter.append_message(
        context=context,  # type: ignore[arg-type]
        message=_message(content="alpha"),
    )
    port = InMemoryThreadMemoryLifecyclePort()
    second_adapter = SessionHistorySnapshotConversationThreadMemoryAdapter(port=port)
    second = second_adapter.append_message(
        context=context,  # type: ignore[arg-type]
        message=_message(content="beta"),
    )
    assert first.snapshot.revision_id != second.snapshot.revision_id


def test_tenant_mismatch_fails_closed() -> None:
    context = _context()
    snapshot = _adapter().append_message(
        context=context,  # type: ignore[arg-type]
        message=_message(),
    )
    tampered = _tamper_envelope(snapshot, tenant_id="tenant-b")
    with pytest.raises(ConversationThreadMemoryError) as exc_info:
        SessionHistorySnapshotConversationThreadMemoryAdapter.load_bounded_history(
            context=context,  # type: ignore[arg-type]
            memory_snapshot=tampered,
            limits=_limits(),
            now=_NOW,
        )
    assert exc_info.value.error_code == "THREAD_MEMORY_TENANT_MISMATCH"


def test_binding_mismatch_fails_closed() -> None:
    context = _context(binding=_BINDING)
    snapshot = _adapter().append_message(
        context=context,  # type: ignore[arg-type]
        message=_message(),
    )
    tampered = _tamper_envelope(snapshot, conversation_context_binding_id=_BINDING_B)
    with pytest.raises(ConversationThreadMemoryError) as exc_info:
        SessionHistorySnapshotConversationThreadMemoryAdapter.load_bounded_history(
            context=context,  # type: ignore[arg-type]
            memory_snapshot=tampered,
            limits=_limits(),
            now=_NOW,
        )
    assert exc_info.value.error_code == "THREAD_MEMORY_BINDING_MISMATCH"


def test_thread_mismatch_fails_closed() -> None:
    context = _context(thread=_THREAD)
    snapshot = _adapter().append_message(
        context=context,  # type: ignore[arg-type]
        message=_message(),
    )
    tampered = _tamper_envelope(snapshot, canonical_thread_ref="tampered-thread")
    with pytest.raises(ConversationThreadMemoryError) as exc_info:
        SessionHistorySnapshotConversationThreadMemoryAdapter.load_bounded_history(
            context=context,  # type: ignore[arg-type]
            memory_snapshot=tampered,
            limits=_limits(),
            now=_NOW,
        )
    assert exc_info.value.error_code == "THREAD_MEMORY_THREAD_MISMATCH"


def test_audience_mismatch_fails_closed() -> None:
    personal = _context(audience=ConversationAudienceMode.PERSONAL)
    snapshot = _adapter().append_message(
        context=personal,  # type: ignore[arg-type]
        message=_message(content="personal"),
    )
    shared = _context(audience=ConversationAudienceMode.SHARED, binding=_BINDING)
    with pytest.raises(ConversationThreadMemoryError) as exc_info:
        SessionHistorySnapshotConversationThreadMemoryAdapter.load_bounded_history(
            context=shared,  # type: ignore[arg-type]
            memory_snapshot=snapshot,
            limits=_limits(),
            now=_NOW,
        )
    assert exc_info.value.error_code == "THREAD_MEMORY_AUDIENCE_MISMATCH"


def test_workspace_mismatch_fails_closed() -> None:
    context = _context()
    snapshot = _adapter().append_message(
        context=context,  # type: ignore[arg-type]
        message=_message(),
    )
    mismatched = _context(workspace=_WORKSPACE_B)
    with pytest.raises(ConversationThreadMemoryError) as exc_info:
        SessionHistorySnapshotConversationThreadMemoryAdapter.load_bounded_history(
            context=mismatched,  # type: ignore[arg-type]
            memory_snapshot=snapshot,
            limits=_limits(),
            now=_NOW,
        )
    assert exc_info.value.error_code == "THREAD_MEMORY_WORKSPACE_MISMATCH"


def test_snapshot_scope_mismatch_fails_closed() -> None:
    context = _context()
    snapshot = _adapter().append_message(
        context=context,  # type: ignore[arg-type]
        message=_message(),
    )
    wrong_scope = _scope(thread="other-thread")
    tampered_snapshot = SessionHistorySnapshot(
        tenant_id=snapshot.snapshot.tenant_id,
        context_scope_id=wrong_scope,
        revision_id=snapshot.snapshot.revision_id,
        messages=snapshot.snapshot.messages,
        source_content_hash=snapshot.snapshot.source_content_hash,
    )
    tampered = _tamper_envelope(snapshot, snapshot=tampered_snapshot)
    with pytest.raises(ConversationThreadMemoryError) as exc_info:
        SessionHistorySnapshotConversationThreadMemoryAdapter.load_bounded_history(
            context=context,  # type: ignore[arg-type]
            memory_snapshot=tampered,
            limits=_limits(),
            now=_NOW,
        )
    assert exc_info.value.error_code == "THREAD_MEMORY_SNAPSHOT_IDENTITY_MISMATCH"


def test_snapshot_tenant_mismatch_fails_closed() -> None:
    context = _context()
    snapshot = _adapter().append_message(
        context=context,  # type: ignore[arg-type]
        message=_message(),
    )
    tampered_snapshot = SessionHistorySnapshot(
        tenant_id="tenant-b",
        context_scope_id=snapshot.snapshot.context_scope_id,
        revision_id=snapshot.snapshot.revision_id,
        messages=snapshot.snapshot.messages,
        source_content_hash=snapshot.snapshot.source_content_hash,
    )
    tampered = _tamper_envelope(snapshot, snapshot=tampered_snapshot)
    with pytest.raises(ConversationThreadMemoryError) as exc_info:
        SessionHistorySnapshotConversationThreadMemoryAdapter.load_bounded_history(
            context=context,  # type: ignore[arg-type]
            memory_snapshot=tampered,
            limits=_limits(),
            now=_NOW,
        )
    assert exc_info.value.error_code == "THREAD_MEMORY_SNAPSHOT_IDENTITY_MISMATCH"


def test_missing_timestamp_fails_closed() -> None:
    context = _context()
    snapshot = _adapter().append_message(
        context=context,  # type: ignore[arg-type]
        message=_message(),
    )
    tampered = _tamper_envelope(snapshot, message_created_at=())
    with pytest.raises(ConversationThreadMemoryError) as exc_info:
        SessionHistorySnapshotConversationThreadMemoryAdapter.load_bounded_history(
            context=context,  # type: ignore[arg-type]
            memory_snapshot=tampered,
            limits=_limits(),
            now=_NOW,
        )
    assert exc_info.value.error_code == "THREAD_MEMORY_CREATED_AT_MISSING"


def test_duplicate_timestamp_id_fails_closed() -> None:
    context = _context()
    snapshot = _adapter().append_message(
        context=context,  # type: ignore[arg-type]
        message=_message(),
    )
    message_id = snapshot.snapshot.messages[0].message_id
    duplicate_entries = (
        (message_id, _NOW.isoformat()),
        (message_id, _NOW.isoformat()),
    )
    tampered = _tamper_envelope(snapshot, message_created_at=duplicate_entries)
    with pytest.raises(ConversationThreadMemoryError) as exc_info:
        SessionHistorySnapshotConversationThreadMemoryAdapter.load_bounded_history(
            context=context,  # type: ignore[arg-type]
            memory_snapshot=tampered,
            limits=_limits(),
            now=_NOW,
        )
    assert exc_info.value.error_code == "THREAD_MEMORY_CREATED_AT_DUPLICATE"


def test_unknown_timestamp_id_fails_closed() -> None:
    context = _context()
    snapshot = _adapter().append_message(
        context=context,  # type: ignore[arg-type]
        message=_message(),
    )
    message_id = snapshot.snapshot.messages[0].message_id
    unknown_entries = (
        (message_id, _NOW.isoformat()),
        ("unknown-message-id", _NOW.isoformat()),
    )
    tampered = _tamper_envelope(snapshot, message_created_at=unknown_entries)
    with pytest.raises(ConversationThreadMemoryError) as exc_info:
        SessionHistorySnapshotConversationThreadMemoryAdapter.load_bounded_history(
            context=context,  # type: ignore[arg-type]
            memory_snapshot=tampered,
            limits=_limits(),
            now=_NOW,
        )
    assert exc_info.value.error_code == "THREAD_MEMORY_CREATED_AT_UNKNOWN"


def test_non_utc_timestamp_fails_closed() -> None:
    context = _context()
    snapshot = _adapter().append_message(
        context=context,  # type: ignore[arg-type]
        message=_message(),
    )
    message_id = snapshot.snapshot.messages[0].message_id
    warsaw = datetime(2024, 6, 1, 12, 0, tzinfo=timezone(timedelta(hours=2)))
    tampered = _tamper_envelope(
        snapshot,
        message_created_at=((message_id, warsaw.isoformat()),),
    )
    with pytest.raises(ConversationThreadMemoryError) as exc_info:
        SessionHistorySnapshotConversationThreadMemoryAdapter.load_bounded_history(
            context=context,  # type: ignore[arg-type]
            memory_snapshot=tampered,
            limits=_limits(),
            now=_NOW,
        )
    assert exc_info.value.error_code == "THREAD_MEMORY_CREATED_AT_INVALID"


def test_two_threads_remain_isolated() -> None:
    first_context = _context(thread=_THREAD)
    second_context = _context(thread=_THREAD_B)
    first_snapshot = _adapter().append_message(
        context=first_context,  # type: ignore[arg-type]
        message=_message(content="first"),
    )
    second_snapshot = _adapter().append_message(
        context=second_context,  # type: ignore[arg-type]
        message=_message(content="second"),
    )
    first_history = SessionHistorySnapshotConversationThreadMemoryAdapter.load_bounded_history(
        context=first_context,  # type: ignore[arg-type]
        memory_snapshot=first_snapshot,
        limits=_limits(),
        now=_NOW,
    )
    second_history = SessionHistorySnapshotConversationThreadMemoryAdapter.load_bounded_history(
        context=second_context,  # type: ignore[arg-type]
        memory_snapshot=second_snapshot,
        limits=_limits(),
        now=_NOW,
    )
    assert first_history[0].content == "first"
    assert second_history[0].content == "second"


def test_two_bindings_remain_isolated() -> None:
    first_context = _context(binding=_BINDING)
    second_context = _context(binding=_BINDING_B)
    first_snapshot = _adapter().append_message(
        context=first_context,  # type: ignore[arg-type]
        message=_message(content="binding-a"),
    )
    second_snapshot = _adapter().append_message(
        context=second_context,  # type: ignore[arg-type]
        message=_message(content="binding-b"),
    )
    first_history = SessionHistorySnapshotConversationThreadMemoryAdapter.load_bounded_history(
        context=first_context,  # type: ignore[arg-type]
        memory_snapshot=first_snapshot,
        limits=_limits(),
        now=_NOW,
    )
    second_history = SessionHistorySnapshotConversationThreadMemoryAdapter.load_bounded_history(
        context=second_context,  # type: ignore[arg-type]
        memory_snapshot=second_snapshot,
        limits=_limits(),
        now=_NOW,
    )
    assert first_history[0].content == "binding-a"
    assert second_history[0].content == "binding-b"


def test_personal_and_shared_partitions_remain_isolated() -> None:
    personal_context = _context(audience=ConversationAudienceMode.PERSONAL)
    shared_context = _context(audience=ConversationAudienceMode.SHARED, binding=_BINDING_B)
    personal_snapshot = _adapter().append_message(
        context=personal_context,  # type: ignore[arg-type]
        message=_message(content="personal"),
    )
    shared_snapshot = _adapter().append_message(
        context=shared_context,  # type: ignore[arg-type]
        message=_message(content="shared"),
    )
    personal_history = SessionHistorySnapshotConversationThreadMemoryAdapter.load_bounded_history(
        context=personal_context,  # type: ignore[arg-type]
        memory_snapshot=personal_snapshot,
        limits=_limits(),
        now=_NOW,
    )
    shared_history = SessionHistorySnapshotConversationThreadMemoryAdapter.load_bounded_history(
        context=shared_context,  # type: ignore[arg-type]
        memory_snapshot=shared_snapshot,
        limits=_limits(),
        now=_NOW,
    )
    assert personal_history[0].content == "personal"
    assert shared_history[0].content == "shared"


def test_max_age_keeps_only_non_expired_messages() -> None:
    context = _context()
    adapter = _adapter()
    snapshot = adapter.append_message(
        context=context,  # type: ignore[arg-type]
        message=_message(content="old", created_at=_NOW - timedelta(seconds=120)),
    )
    snapshot = adapter.append_message(
        context=context,  # type: ignore[arg-type]
        message=_message(content="fresh", created_at=_NOW - timedelta(seconds=10)),
    )
    history = SessionHistorySnapshotConversationThreadMemoryAdapter.load_bounded_history(
        context=context,  # type: ignore[arg-type]
        memory_snapshot=snapshot,
        limits=_limits(max_age_seconds=60),
        now=_NOW,
    )
    assert [item.content for item in history] == ["fresh"]


def test_max_messages_retains_newest_suffix() -> None:
    context = _context()
    adapter = _adapter()
    snapshot = None
    for index in range(4):
        snapshot = adapter.append_message(
            context=context,  # type: ignore[arg-type]
                message=_message(content=f"m{index}", created_at=_NOW + timedelta(seconds=index)),
        )
    history = SessionHistorySnapshotConversationThreadMemoryAdapter.load_bounded_history(
        context=context,  # type: ignore[arg-type]
        memory_snapshot=snapshot,
        limits=_limits(max_messages=2),
        now=_NOW + timedelta(seconds=10),
    )
    assert [item.content for item in history] == ["m2", "m3"]


def test_max_bytes_retains_newest_suffix() -> None:
    context = _context()
    adapter = _adapter()
    snapshot = adapter.append_message(
        context=context,  # type: ignore[arg-type]
        message=_message(content="aaaa", created_at=_NOW),
    )
    snapshot = adapter.append_message(
        context=context,  # type: ignore[arg-type]
        message=_message(content="bbbb", created_at=_NOW + timedelta(seconds=1)),
    )
    history = SessionHistorySnapshotConversationThreadMemoryAdapter.load_bounded_history(
        context=context,  # type: ignore[arg-type]
        memory_snapshot=snapshot,
        limits=_limits(max_bytes=5),
        now=_NOW + timedelta(seconds=2),
    )
    assert [item.content for item in history] == ["bbbb"]


def test_oversized_newest_message_returns_empty_tuple() -> None:
    context = _context()
    snapshot = _adapter().append_message(
        context=context,  # type: ignore[arg-type]
        message=_message(content="x" * 20, created_at=_NOW),
    )
    history = SessionHistorySnapshotConversationThreadMemoryAdapter.load_bounded_history(
        context=context,  # type: ignore[arg-type]
        memory_snapshot=snapshot,
        limits=_limits(max_bytes=10),
        now=_NOW,
    )
    assert history == ()


def test_returned_history_is_immutable() -> None:
    context = _context()
    snapshot = _adapter().append_message(
        context=context,  # type: ignore[arg-type]
        message=_message(),
    )
    history = SessionHistorySnapshotConversationThreadMemoryAdapter.load_bounded_history(
        context=context,  # type: ignore[arg-type]
        memory_snapshot=snapshot,
        limits=_limits(),
        now=_NOW,
    )
    with pytest.raises((TypeError, ValueError)):
        history[0].content = "mutated"  # type: ignore[misc]


def test_no_persistence_network_or_provider_calls() -> None:
    context = _context()
    snapshot = _adapter().append_message(
        context=context,  # type: ignore[arg-type]
        message=_message(content="pure"),
    )
    history = SessionHistorySnapshotConversationThreadMemoryAdapter.load_bounded_history(
        context=context,  # type: ignore[arg-type]
        memory_snapshot=snapshot,
        limits=_limits(),
        now=_NOW,
    )
    assert history[0].content == "pure"


def test_same_identity_produces_same_session_key() -> None:
    first = derive_conversation_thread_session_key(
        tenant_id=_TENANT,
        conversation_context_binding_id=_BINDING,
        canonical_thread_ref=_THREAD,
    )
    second = derive_conversation_thread_session_key(
        tenant_id=_TENANT,
        conversation_context_binding_id=_BINDING,
        canonical_thread_ref=_THREAD,
    )
    assert first == second
    assert first.startswith("lkw-conversation-thread:v1:")


def test_partition_descriptor_fields() -> None:
    context = _context()
    partition = ConversationThreadMemoryPartitionV1.from_execution_context(context)  # type: ignore[arg-type]
    assert partition.tenant_id == _TENANT
    assert partition.conversation_context_binding_id == _BINDING
    assert partition.canonical_thread_ref == _THREAD
    assert partition.audience_mode is ConversationAudienceMode.PERSONAL
    assert partition.workspace_id == _WORKSPACE
