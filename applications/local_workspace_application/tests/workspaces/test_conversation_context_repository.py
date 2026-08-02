# © Artur Czarnecki. All rights reserved.

"""Unit tests for Conversation Context repository persistence."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.integrations.contracts.document_store import DocumentRecord
from local_workspace_application.workspaces.conversation_context_models import (
    ConversationActivationPolicy,
    ConversationAudienceMode,
    ConversationContextBindingStatus,
    ConversationContextBindingV1,
    ConversationThreadContextPolicy,
    ConversationWorkspaceResolutionPolicy,
    PersonalConversationStateV1,
    WorkspaceConversationAudience,
    WorkspaceConversationAudiencePolicyV1,
)
from local_workspace_application.workspaces.conversation_context_repository import (
    ConversationContextRepository,
    ConversationContextRepositoryError,
    _partition,
)

pytestmark = pytest.mark.unit

_NOW = datetime(2024, 6, 1, 12, 0, tzinfo=UTC)
_TENANT = "tenant-a"
_TENANT_B = "tenant-b"
_BINDING_ID = "binding-1"
_BINDING_ID_B = "binding-2"
_CONNECTION = "conn.primary"
_CONVERSATION = "conv.alpha"
_PRINCIPAL = "principal.alice"
_WORKSPACE = "workspace-1"
_WORKSPACE_B = "workspace-2"
_ENTITY_BINDING = "conversation_context_binding"


def _repo(store: InMemoryDocumentStore | None = None) -> ConversationContextRepository:
    return ConversationContextRepository(store or InMemoryDocumentStore())


def _binding(**overrides: object) -> ConversationContextBindingV1:
    payload = {
        "conversation_context_binding_id": _BINDING_ID,
        "tenant_id": _TENANT,
        "conversation_connection_ref": _CONNECTION,
        "frontend_provider_id": "provider.web",
        "opaque_conversation_ref": _CONVERSATION,
        "audience_mode": ConversationAudienceMode.PERSONAL,
        "workspace_resolution_policy": ConversationWorkspaceResolutionPolicy.FIXED_WORKSPACE,
        "workspace_id": _WORKSPACE,
        "owner_principal_ref": _PRINCIPAL,
        "activation_policy": ConversationActivationPolicy.ALWAYS,
        "thread_context_policy": ConversationThreadContextPolicy.CURRENT_THREAD_BOUNDED,
        "administrative_status": ConversationContextBindingStatus.ACTIVE,
        "configuration_version": 1,
        "created_at": _NOW,
        "updated_at": _NOW,
    }
    payload.update(overrides)
    return ConversationContextBindingV1(**payload)  # type: ignore[arg-type]


def test_binding_survives_second_repository_instance() -> None:
    store = InMemoryDocumentStore()
    binding = _binding()
    assert _repo(store).put_binding_if_absent(binding) is True
    loaded = _repo(store).get_binding(
        tenant_id=_TENANT,
        conversation_connection_ref=_CONNECTION,
        opaque_conversation_ref=_CONVERSATION,
        conversation_context_binding_id=_BINDING_ID,
    )
    assert loaded == binding


def test_semantic_identity_is_tenant_connection_conversation() -> None:
    repo = _repo()
    binding_a = _binding(conversation_context_binding_id="binding-a")
    binding_b = _binding(
        conversation_context_binding_id="binding-b",
        opaque_conversation_ref="conv.other",
    )
    assert repo.put_binding_if_absent(binding_a) is True
    assert repo.put_binding_if_absent(binding_b) is True
    listed = repo.list_bindings_for_semantic_identity(
        tenant_id=_TENANT,
        conversation_connection_ref=_CONNECTION,
        opaque_conversation_ref=_CONVERSATION,
    )
    assert [item.conversation_context_binding_id for item in listed] == ["binding-a"]


def test_tenant_isolation() -> None:
    store = InMemoryDocumentStore()
    repo_a = _repo(store)
    repo_b = _repo(store)
    assert repo_a.put_binding_if_absent(_binding()) is True
    assert (
        repo_b.get_binding(
            tenant_id=_TENANT_B,
            conversation_connection_ref=_CONNECTION,
            opaque_conversation_ref=_CONVERSATION,
            conversation_context_binding_id=_BINDING_ID,
        )
        is None
    )
    assert repo_b.list_bindings_for_semantic_identity(
        tenant_id=_TENANT_B,
        conversation_connection_ref=_CONNECTION,
        opaque_conversation_ref=_CONVERSATION,
    ) == []


def test_binding_cas_succeeds_once() -> None:
    repo = _repo()
    original = _binding()
    assert repo.put_binding_if_absent(original) is True
    updated = original.model_copy(update={"configuration_version": 2, "updated_at": _NOW})
    assert repo.replace_binding_if_match(expected=original, replacement=updated) is True
    loaded = repo.get_binding(
        tenant_id=_TENANT,
        conversation_connection_ref=_CONNECTION,
        opaque_conversation_ref=_CONVERSATION,
        conversation_context_binding_id=_BINDING_ID,
    )
    assert loaded is not None
    assert loaded.configuration_version == 2


def test_stale_binding_cas_fails() -> None:
    repo = _repo()
    original = _binding()
    assert repo.put_binding_if_absent(original) is True
    updated = original.model_copy(update={"configuration_version": 2, "updated_at": _NOW})
    assert repo.replace_binding_if_match(expected=original, replacement=updated) is True
    assert repo.replace_binding_if_match(expected=original, replacement=updated) is False


def test_malformed_record_fails_with_deterministic_repository_error() -> None:
    store = InMemoryDocumentStore()
    partition_key = _partition(_TENANT, _ENTITY_BINDING)
    store.put(
        DocumentRecord(
            partition_key=partition_key,
            row_key=f"{_CONNECTION}\x1e{_CONVERSATION}\x1e{_BINDING_ID}",
            data={"conversation_context_binding_id": _BINDING_ID},
        )
    )
    repo = _repo(store)
    with pytest.raises(ConversationContextRepositoryError) as exc_info:
        repo.list_bindings_for_semantic_identity(
            tenant_id=_TENANT,
            conversation_connection_ref=_CONNECTION,
            opaque_conversation_ref=_CONVERSATION,
        )
    assert exc_info.value.error_code == "conversation_context_malformed_record"


def test_personal_state_isolated_by_binding_and_principal() -> None:
    repo = _repo()
    state_a = PersonalConversationStateV1(
        tenant_id=_TENANT,
        conversation_context_binding_id=_BINDING_ID,
        owner_principal_ref=_PRINCIPAL,
        selected_workspace_id=_WORKSPACE,
        configuration_version=1,
        updated_at=_NOW,
    )
    state_b = PersonalConversationStateV1(
        tenant_id=_TENANT,
        conversation_context_binding_id=_BINDING_ID_B,
        owner_principal_ref=_PRINCIPAL,
        selected_workspace_id=_WORKSPACE_B,
        configuration_version=1,
        updated_at=_NOW,
    )
    assert repo.put_personal_state_if_absent(state_a) is True
    assert repo.put_personal_state_if_absent(state_b) is True
    loaded_a = repo.get_personal_state(
        tenant_id=_TENANT,
        conversation_context_binding_id=_BINDING_ID,
        owner_principal_ref=_PRINCIPAL,
    )
    loaded_b = repo.get_personal_state(
        tenant_id=_TENANT,
        conversation_context_binding_id=_BINDING_ID_B,
        owner_principal_ref=_PRINCIPAL,
    )
    assert loaded_a is not None and loaded_a.selected_workspace_id == _WORKSPACE
    assert loaded_b is not None and loaded_b.selected_workspace_id == _WORKSPACE_B


def test_workspace_audience_policy_is_tenant_workspace_scoped() -> None:
    store = InMemoryDocumentStore()
    repo_a = _repo(store)
    repo_b = _repo(store)
    policy = WorkspaceConversationAudiencePolicyV1(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        audience=WorkspaceConversationAudience.PERSONAL,
        configuration_version=1,
        updated_at=_NOW,
    )
    assert repo_a.put_workspace_audience_policy_if_absent(policy) is True
    assert repo_b.get_workspace_audience_policy(tenant_id=_TENANT_B, workspace_id=_WORKSPACE) is None
    loaded = repo_a.get_workspace_audience_policy(tenant_id=_TENANT, workspace_id=_WORKSPACE)
    assert loaded == policy
