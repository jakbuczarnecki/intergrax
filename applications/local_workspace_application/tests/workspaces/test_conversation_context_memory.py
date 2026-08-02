# © Artur Czarnecki. All rights reserved.

"""Unit tests for Conversation thread memory adapter."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from local_workspace_application.workspaces.conversation_context_execution import (
    build_conversation_execution_context,
)
from local_workspace_application.workspaces.conversation_context_memory import (
    ContextLifecycleConversationThreadMemoryAdapter,
    ConversationThreadMemoryError,
    ConversationThreadMemoryPartitionV1,
    ThreadMemoryLifecycleEnvelopeV1,
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


class FakeContextLifecycleThreadMemoryPort:
    def __init__(self) -> None:
        self._store: dict[str, ThreadMemoryLifecycleEnvelopeV1] = {}
        self.load_calls = 0
        self.save_calls = 0

    def load_envelope(
        self,
        *,
        tenant_id: str,
        context_scope_id: str,
    ) -> ThreadMemoryLifecycleEnvelopeV1 | None:
        self.load_calls += 1
        return self._store.get(context_scope_id)

    def save_envelope(self, *, envelope: ThreadMemoryLifecycleEnvelopeV1) -> None:
        self.save_calls += 1
        self._store[envelope.snapshot.context_scope_id] = envelope


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


def test_changing_tenant_changes_session_key() -> None:
    base = derive_conversation_thread_session_key(
        tenant_id=_TENANT,
        conversation_context_binding_id=_BINDING,
        canonical_thread_ref=_THREAD,
    )
    changed = derive_conversation_thread_session_key(
        tenant_id="tenant-b",
        conversation_context_binding_id=_BINDING,
        canonical_thread_ref=_THREAD,
    )
    assert base != changed


def test_changing_binding_changes_session_key() -> None:
    base = derive_conversation_thread_session_key(
        tenant_id=_TENANT,
        conversation_context_binding_id=_BINDING,
        canonical_thread_ref=_THREAD,
    )
    changed = derive_conversation_thread_session_key(
        tenant_id=_TENANT,
        conversation_context_binding_id=_BINDING_B,
        canonical_thread_ref=_THREAD,
    )
    assert base != changed


def test_changing_thread_changes_session_key() -> None:
    base = derive_conversation_thread_session_key(
        tenant_id=_TENANT,
        conversation_context_binding_id=_BINDING,
        canonical_thread_ref=_THREAD,
    )
    changed = derive_conversation_thread_session_key(
        tenant_id=_TENANT,
        conversation_context_binding_id=_BINDING,
        canonical_thread_ref=_THREAD_B,
    )
    assert base != changed


def test_personal_and_shared_histories_do_not_mix() -> None:
    port = FakeContextLifecycleThreadMemoryPort()
    adapter = ContextLifecycleConversationThreadMemoryAdapter(port)
    personal = _context(audience=ConversationAudienceMode.PERSONAL, binding=_BINDING)
    shared = _context(audience=ConversationAudienceMode.SHARED, binding=_BINDING_B)
    adapter.append_message(context=personal, message=_message(content="personal"))
    adapter.append_message(context=shared, message=_message(content="shared"))
    personal_history = adapter.load_bounded_history(context=personal, limits=_limits(), now=_NOW)
    shared_history = adapter.load_bounded_history(context=shared, limits=_limits(), now=_NOW)
    assert len(personal_history) == 1
    assert personal_history[0].content == "personal"
    assert len(shared_history) == 1
    assert shared_history[0].content == "shared"


def test_two_thread_refs_do_not_mix() -> None:
    port = FakeContextLifecycleThreadMemoryPort()
    adapter = ContextLifecycleConversationThreadMemoryAdapter(port)
    first = _context(thread=_THREAD)
    second = _context(thread=_THREAD_B)
    adapter.append_message(context=first, message=_message(content="first"))
    adapter.append_message(context=second, message=_message(content="second"))
    assert adapter.load_bounded_history(context=first, limits=_limits(), now=_NOW)[0].content == "first"
    assert adapter.load_bounded_history(context=second, limits=_limits(), now=_NOW)[0].content == "second"


def test_two_bindings_do_not_mix() -> None:
    port = FakeContextLifecycleThreadMemoryPort()
    adapter = ContextLifecycleConversationThreadMemoryAdapter(port)
    first = _context(binding=_BINDING)
    second = _context(binding=_BINDING_B)
    adapter.append_message(context=first, message=_message(content="binding-a"))
    adapter.append_message(context=second, message=_message(content="binding-b"))
    assert adapter.load_bounded_history(context=first, limits=_limits(), now=_NOW)[0].content == "binding-a"
    assert adapter.load_bounded_history(context=second, limits=_limits(), now=_NOW)[0].content == "binding-b"


def test_workspace_mismatch_fails_closed() -> None:
    port = FakeContextLifecycleThreadMemoryPort()
    adapter = ContextLifecycleConversationThreadMemoryAdapter(port)
    context = _context()
    adapter.append_message(context=context, message=_message())
    mismatched = _context(workspace=_WORKSPACE_B)
    with pytest.raises(ConversationThreadMemoryError) as exc_info:
        adapter.load_bounded_history(context=mismatched, limits=_limits(), now=_NOW)
    assert exc_info.value.error_code == "THREAD_MEMORY_WORKSPACE_MISMATCH"


def test_audience_mismatch_fails_closed() -> None:
    port = FakeContextLifecycleThreadMemoryPort()
    adapter = ContextLifecycleConversationThreadMemoryAdapter(port)
    personal = _context(audience=ConversationAudienceMode.PERSONAL)
    adapter.append_message(context=personal, message=_message(content="personal"))
    shared = _context(audience=ConversationAudienceMode.SHARED)
    with pytest.raises(ConversationThreadMemoryError) as exc_info:
        adapter.load_bounded_history(context=shared, limits=_limits(), now=_NOW)
    assert exc_info.value.error_code == "THREAD_MEMORY_AUDIENCE_MISMATCH"


def test_tenant_mismatch_fails_closed() -> None:
    port = FakeContextLifecycleThreadMemoryPort()
    adapter = ContextLifecycleConversationThreadMemoryAdapter(port)
    context = _context()
    adapter.append_message(context=context, message=_message())
    scope = derive_conversation_thread_session_key(
        tenant_id=_TENANT,
        conversation_context_binding_id=_BINDING,
        canonical_thread_ref=_THREAD,
    )
    stored = port.load_envelope(tenant_id=_TENANT, context_scope_id=scope)
    assert stored is not None
    tampered = ThreadMemoryLifecycleEnvelopeV1(
        schema_version=stored.schema_version,
        tenant_id="tenant-b",
        conversation_context_binding_id=stored.conversation_context_binding_id,
        canonical_thread_ref=stored.canonical_thread_ref,
        audience_mode=stored.audience_mode,
        workspace_id=stored.workspace_id,
        snapshot=stored.snapshot,
        message_created_at=stored.message_created_at,
    )
    port.save_envelope(envelope=tampered)
    with pytest.raises(ConversationThreadMemoryError) as exc_info:
        adapter.load_bounded_history(context=context, limits=_limits(), now=_NOW)
    assert exc_info.value.error_code == "THREAD_MEMORY_TENANT_MISMATCH"


def test_binding_mismatch_fails_closed() -> None:
    port = FakeContextLifecycleThreadMemoryPort()
    adapter = ContextLifecycleConversationThreadMemoryAdapter(port)
    context = _context(binding=_BINDING)
    adapter.append_message(context=context, message=_message())
    scope = derive_conversation_thread_session_key(
        tenant_id=_TENANT,
        conversation_context_binding_id=_BINDING,
        canonical_thread_ref=_THREAD,
    )
    stored = port.load_envelope(tenant_id=_TENANT, context_scope_id=scope)
    assert stored is not None
    tampered = ThreadMemoryLifecycleEnvelopeV1(
        schema_version=stored.schema_version,
        tenant_id=stored.tenant_id,
        conversation_context_binding_id=_BINDING_B,
        canonical_thread_ref=stored.canonical_thread_ref,
        audience_mode=stored.audience_mode,
        workspace_id=stored.workspace_id,
        snapshot=stored.snapshot,
        message_created_at=stored.message_created_at,
    )
    port.save_envelope(envelope=tampered)
    with pytest.raises(ConversationThreadMemoryError) as exc_info:
        adapter.load_bounded_history(context=context, limits=_limits(), now=_NOW)
    assert exc_info.value.error_code == "THREAD_MEMORY_BINDING_MISMATCH"


def test_thread_mismatch_fails_closed() -> None:
    port = FakeContextLifecycleThreadMemoryPort()
    adapter = ContextLifecycleConversationThreadMemoryAdapter(port)
    context = _context(thread=_THREAD)
    adapter.append_message(context=context, message=_message())
    scope = derive_conversation_thread_session_key(
        tenant_id=_TENANT,
        conversation_context_binding_id=_BINDING,
        canonical_thread_ref=_THREAD,
    )
    stored = port.load_envelope(tenant_id=_TENANT, context_scope_id=scope)
    assert stored is not None
    tampered = ThreadMemoryLifecycleEnvelopeV1(
        schema_version=stored.schema_version,
        tenant_id=stored.tenant_id,
        conversation_context_binding_id=stored.conversation_context_binding_id,
        canonical_thread_ref="tampered-thread",
        audience_mode=stored.audience_mode,
        workspace_id=stored.workspace_id,
        snapshot=stored.snapshot,
        message_created_at=stored.message_created_at,
    )
    port.save_envelope(envelope=tampered)
    with pytest.raises(ConversationThreadMemoryError) as exc_info:
        adapter.load_bounded_history(context=context, limits=_limits(), now=_NOW)
    assert exc_info.value.error_code == "THREAD_MEMORY_THREAD_MISMATCH"


def test_chronological_order_preserved() -> None:
    port = FakeContextLifecycleThreadMemoryPort()
    adapter = ContextLifecycleConversationThreadMemoryAdapter(port)
    context = _context()
    adapter.append_message(
        context=context,
        message=_message(content="one", created_at=_NOW - timedelta(minutes=2)),
    )
    adapter.append_message(
        context=context,
        message=_message(content="two", created_at=_NOW - timedelta(minutes=1)),
    )
    history = adapter.load_bounded_history(context=context, limits=_limits(), now=_NOW)
    assert [item.content for item in history] == ["one", "two"]


def test_max_age_seconds_removes_expired_entries() -> None:
    port = FakeContextLifecycleThreadMemoryPort()
    adapter = ContextLifecycleConversationThreadMemoryAdapter(port)
    context = _context()
    adapter.append_message(
        context=context,
        message=_message(content="old", created_at=_NOW - timedelta(seconds=120)),
    )
    adapter.append_message(
        context=context,
        message=_message(content="fresh", created_at=_NOW - timedelta(seconds=10)),
    )
    history = adapter.load_bounded_history(
        context=context,
        limits=_limits(max_age_seconds=60),
        now=_NOW,
    )
    assert [item.content for item in history] == ["fresh"]


def test_max_messages_keeps_newest_suffix() -> None:
    port = FakeContextLifecycleThreadMemoryPort()
    adapter = ContextLifecycleConversationThreadMemoryAdapter(port)
    context = _context()
    for index in range(4):
        adapter.append_message(
            context=context,
            message=_message(content=f"m{index}", created_at=_NOW + timedelta(seconds=index)),
        )
    history = adapter.load_bounded_history(
        context=context,
        limits=_limits(max_messages=2),
        now=_NOW + timedelta(seconds=10),
    )
    assert [item.content for item in history] == ["m2", "m3"]


def test_max_bytes_keeps_newest_suffix() -> None:
    port = FakeContextLifecycleThreadMemoryPort()
    adapter = ContextLifecycleConversationThreadMemoryAdapter(port)
    context = _context()
    adapter.append_message(
        context=context,
        message=_message(content="aaaa", created_at=_NOW),
    )
    adapter.append_message(
        context=context,
        message=_message(content="bbbb", created_at=_NOW + timedelta(seconds=1)),
    )
    history = adapter.load_bounded_history(
        context=context,
        limits=_limits(max_bytes=5),
        now=_NOW + timedelta(seconds=2),
    )
    assert [item.content for item in history] == ["bbbb"]


def test_oversized_newest_message_returns_empty_tuple() -> None:
    port = FakeContextLifecycleThreadMemoryPort()
    adapter = ContextLifecycleConversationThreadMemoryAdapter(port)
    context = _context()
    adapter.append_message(
        context=context,
        message=_message(content="x" * 20, created_at=_NOW),
    )
    history = adapter.load_bounded_history(
        context=context,
        limits=_limits(max_bytes=10),
        now=_NOW,
    )
    assert history == ()


def test_append_uses_platform_lifecycle_port() -> None:
    port = FakeContextLifecycleThreadMemoryPort()
    adapter = ContextLifecycleConversationThreadMemoryAdapter(port)
    context = _context()
    adapter.append_message(context=context, message=_message(content="stored"))
    assert port.save_calls == 1
    assert port.load_calls >= 1


def test_append_exchange_preserves_user_then_assistant_order() -> None:
    port = FakeContextLifecycleThreadMemoryPort()
    adapter = ContextLifecycleConversationThreadMemoryAdapter(port)
    context = _context()
    adapter.append_exchange(
        context=context,
        user_message=_message(
            role=ConversationThreadMemoryMessageRole.USER,
            content="question",
            created_at=_NOW,
        ),
        assistant_message=_message(
            role=ConversationThreadMemoryMessageRole.ASSISTANT,
            content="answer",
            created_at=_NOW + timedelta(seconds=1),
        ),
    )
    history = adapter.load_bounded_history(context=context, limits=_limits(), now=_NOW + timedelta(seconds=2))
    assert [item.role for item in history] == [
        ConversationThreadMemoryMessageRole.USER,
        ConversationThreadMemoryMessageRole.ASSISTANT,
    ]


def test_history_returned_from_adapter_is_immutable() -> None:
    port = FakeContextLifecycleThreadMemoryPort()
    adapter = ContextLifecycleConversationThreadMemoryAdapter(port)
    context = _context()
    adapter.append_message(context=context, message=_message())
    history = adapter.load_bounded_history(context=context, limits=_limits(), now=_NOW)
    with pytest.raises((TypeError, ValueError)):
        history[0].content = "mutated"  # type: ignore[misc]


def test_adapter_construction_performs_no_persistence_or_network_work() -> None:
    port = FakeContextLifecycleThreadMemoryPort()
    ContextLifecycleConversationThreadMemoryAdapter(port)
    assert port.load_calls == 0
    assert port.save_calls == 0


def test_partition_descriptor_fields() -> None:
    context = _context()
    partition = ConversationThreadMemoryPartitionV1.from_execution_context(context)  # type: ignore[arg-type]
    assert partition.tenant_id == _TENANT
    assert partition.conversation_context_binding_id == _BINDING
    assert partition.canonical_thread_ref == _THREAD
    assert partition.audience_mode is ConversationAudienceMode.PERSONAL
    assert partition.workspace_id == _WORKSPACE
