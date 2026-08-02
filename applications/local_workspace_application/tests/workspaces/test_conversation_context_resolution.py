# © Artur Czarnecki. All rights reserved.

"""Unit tests for deterministic Conversation Context resolution."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

import pytest

from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.integrations.contracts.document_store import DocumentRecord
from local_workspace_application.workspaces.conversation_context_models import (
    ConversationActivationPolicy,
    ConversationActivationSignal,
    ConversationAudienceMode,
    ConversationContextBindingStatus,
    ConversationContextBindingV1,
    ConversationIngressContextV1,
    ConversationObservedAudience,
    ConversationThreadContextPolicy,
    ConversationWorkspaceResolutionPolicy,
    PersonalConversationStateV1,
    WorkspaceConversationAudience,
    WorkspaceConversationAudiencePolicyV1,
)
from local_workspace_application.workspaces.conversation_context_repository import (
    ConversationContextRepository,
    ConversationContextRepositoryError,
)
from local_workspace_application.workspaces.conversation_context_resolution import (
    ConversationContextResolutionError,
    ConversationContextResolver,
)

pytestmark = pytest.mark.unit

_NOW = datetime(2024, 6, 1, 12, 0, tzinfo=UTC)
_TENANT = "tenant-a"
_TENANT_B = "tenant-b"
_BINDING_ID = "binding-1"
_CONNECTION = "conn.primary"
_CONVERSATION = "conv.alpha"
_PRINCIPAL = "principal.alice"
_OTHER_PRINCIPAL = "principal.bob"
_WORKSPACE = "workspace-1"
_WORKSPACE_SHARED = "workspace-shared"
_THREAD = "thread-canonical-1"


@dataclass
class _ConnectionPort:
    active_connections: set[tuple[str, str]] = field(default_factory=set)
    calls: list[str] = field(default_factory=list)

    def is_conversation_connection_active_and_tenant_owned(
        self,
        *,
        tenant_id: str,
        conversation_connection_ref: str,
    ) -> bool:
        self.calls.append("connection")
        return (tenant_id, conversation_connection_ref) in self.active_connections


@dataclass
class _WorkspacePort:
    active_workspaces: set[tuple[str, str]] = field(default_factory=set)
    authorized: set[tuple[str, str, str]] = field(default_factory=set)
    calls: list[str] = field(default_factory=list)

    def is_workspace_active(self, *, tenant_id: str, workspace_id: str) -> bool:
        self.calls.append("workspace_active")
        return (tenant_id, workspace_id) in self.active_workspaces

    def may_principal_use_workspace(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        principal_ref: str,
    ) -> bool:
        self.calls.append("workspace_authorize")
        return (tenant_id, workspace_id, principal_ref) in self.authorized


class _TrackingRepository(ConversationContextRepository):
    def __init__(self, store: InMemoryDocumentStore) -> None:
        super().__init__(store)
        self.personal_state_reads = 0

    def get_personal_state(
        self,
        *,
        tenant_id: str,
        conversation_context_binding_id: str,
        owner_principal_ref: str,
    ):
        self.personal_state_reads += 1
        return super().get_personal_state(
            tenant_id=tenant_id,
            conversation_context_binding_id=conversation_context_binding_id,
            owner_principal_ref=owner_principal_ref,
        )


def _ingress(**overrides: object) -> ConversationIngressContextV1:
    payload = {
        "conversation_connection_ref": _CONNECTION,
        "opaque_conversation_ref": _CONVERSATION,
        "opaque_thread_ref": _THREAD,
        "actor_principal_ref": _PRINCIPAL,
        "observed_audience": ConversationObservedAudience.PERSONAL,
        "activation_signal": ConversationActivationSignal.ORDINARY_MESSAGE,
        "provider_event_ref": "evt-1",
    }
    payload.update(overrides)
    return ConversationIngressContextV1(**payload)  # type: ignore[arg-type]


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


def _resolver(
  repository: ConversationContextRepository | None = None,
  *,
  connection_port: _ConnectionPort | None = None,
  workspace_port: _WorkspacePort | None = None,
) -> tuple[ConversationContextResolver, ConversationContextRepository, _ConnectionPort, _WorkspacePort]:
    repo = repository or ConversationContextRepository(InMemoryDocumentStore())
    connection = connection_port or _ConnectionPort()
    workspace = workspace_port or _WorkspacePort()
    return (
        ConversationContextResolver(
            repo,
            connection_port=connection,
            workspace_port=workspace,
        ),
        repo,
        connection,
        workspace,
    )


def _seed_workspace_context(
    repo: ConversationContextRepository,
    workspace_port: _WorkspacePort,
    connection_port: _ConnectionPort,
    *,
    workspace_id: str = _WORKSPACE,
    audience: WorkspaceConversationAudience = WorkspaceConversationAudience.PERSONAL,
) -> None:
    connection_port.active_connections.add((_TENANT, _CONNECTION))
    workspace_port.active_workspaces.add((_TENANT, workspace_id))
    workspace_port.authorized.add((_TENANT, workspace_id, _PRINCIPAL))
    assert (
        repo.put_workspace_audience_policy_if_absent(
            WorkspaceConversationAudiencePolicyV1(
                tenant_id=_TENANT,
                workspace_id=workspace_id,
                audience=audience,
                configuration_version=1,
                updated_at=_NOW,
            )
        )
        is True
    )


def _seed_personal_fixed(
    repo: ConversationContextRepository,
    workspace_port: _WorkspacePort,
    connection_port: _ConnectionPort,
    *,
    audience: WorkspaceConversationAudience = WorkspaceConversationAudience.PERSONAL,
) -> None:
    assert repo.put_binding_if_absent(_binding()) is True
    _seed_workspace_context(repo, workspace_port, connection_port, audience=audience)


def test_unknown_fails_before_repository_or_authorization() -> None:
    resolver, repo, connection, workspace = _resolver()
    with pytest.raises(ConversationContextResolutionError) as exc_info:
        resolver.resolve(
            tenant_id=_TENANT,
            ingress=_ingress(observed_audience=ConversationObservedAudience.UNKNOWN),
        )
    assert exc_info.value.error_code == "OBSERVED_AUDIENCE_UNKNOWN"
    assert connection.calls == []
    assert workspace.calls == []
    assert repo.list_bindings_for_semantic_identity(
        tenant_id=_TENANT,
        conversation_connection_ref=_CONNECTION,
        opaque_conversation_ref=_CONVERSATION,
    ) == []


def test_no_active_binding_fails() -> None:
    resolver, _, connection, workspace = _resolver()
    with pytest.raises(ConversationContextResolutionError) as exc_info:
        resolver.resolve(tenant_id=_TENANT, ingress=_ingress())
    assert exc_info.value.error_code == "NO_ACTIVE_BINDING"
    assert connection.calls == []
    assert workspace.calls == []


def test_two_active_bindings_fail() -> None:
    repo = ConversationContextRepository(InMemoryDocumentStore())
    assert repo.put_binding_if_absent(_binding(conversation_context_binding_id="binding-a")) is True
    assert repo.put_binding_if_absent(_binding(conversation_context_binding_id="binding-b")) is True
    resolver, _, connection, workspace = _resolver(repo)
    with pytest.raises(ConversationContextResolutionError) as exc_info:
        resolver.resolve(tenant_id=_TENANT, ingress=_ingress())
    assert exc_info.value.error_code == "AMBIGUOUS_ACTIVE_BINDING"
    assert connection.calls == []
    assert workspace.calls == []


def test_disabled_binding_is_ignored() -> None:
    repo = ConversationContextRepository(InMemoryDocumentStore())
    assert (
        repo.put_binding_if_absent(
            _binding(
                administrative_status=ConversationContextBindingStatus.DISABLED,
            )
        )
        is True
    )
    resolver, _, connection, workspace = _resolver(repo)
    with pytest.raises(ConversationContextResolutionError) as exc_info:
        resolver.resolve(tenant_id=_TENANT, ingress=_ingress())
    assert exc_info.value.error_code == "NO_ACTIVE_BINDING"
    assert connection.calls == []
    assert workspace.calls == []


def test_inactive_connection_fails_before_workspace_resolution() -> None:
    repo = ConversationContextRepository(InMemoryDocumentStore())
    assert repo.put_binding_if_absent(_binding()) is True
    resolver, _, connection, workspace = _resolver(repo)
    with pytest.raises(ConversationContextResolutionError) as exc_info:
        resolver.resolve(tenant_id=_TENANT, ingress=_ingress())
    assert exc_info.value.error_code == "CONVERSATION_CONNECTION_UNAVAILABLE"
    assert connection.calls == ["connection"]
    assert workspace.calls == []


def test_audience_mismatch_fails_before_workspace_resolution() -> None:
    repo = ConversationContextRepository(InMemoryDocumentStore())
    assert repo.put_binding_if_absent(_binding()) is True
    connection = _ConnectionPort(active_connections={(_TENANT, _CONNECTION)})
    workspace = _WorkspacePort()
    resolver, _, connection, workspace = _resolver(repo, connection_port=connection, workspace_port=workspace)
    with pytest.raises(ConversationContextResolutionError) as exc_info:
        resolver.resolve(
            tenant_id=_TENANT,
            ingress=_ingress(observed_audience=ConversationObservedAudience.SHARED),
        )
    assert exc_info.value.error_code == "AUDIENCE_MISMATCH"
    assert workspace.calls == []


def test_personal_principal_mismatch_fails() -> None:
    repo = ConversationContextRepository(InMemoryDocumentStore())
    connection = _ConnectionPort()
    workspace = _WorkspacePort()
    _seed_personal_fixed(repo, workspace, connection)
    resolver, _, _, _ = _resolver(repo, connection_port=connection, workspace_port=workspace)
    with pytest.raises(ConversationContextResolutionError) as exc_info:
        resolver.resolve(
            tenant_id=_TENANT,
            ingress=_ingress(actor_principal_ref=_OTHER_PRINCIPAL),
        )
    assert exc_info.value.error_code == "PERSONAL_PRINCIPAL_MISMATCH"


def test_always_accepts_ordinary_message() -> None:
    repo = ConversationContextRepository(InMemoryDocumentStore())
    connection = _ConnectionPort(active_connections={(_TENANT, _CONNECTION)})
    workspace = _WorkspacePort(
        active_workspaces={(_TENANT, _WORKSPACE)},
        authorized={(_TENANT, _WORKSPACE, _PRINCIPAL)},
    )
    _seed_personal_fixed(repo, workspace, connection)
    resolver, _, _, _ = _resolver(repo, connection_port=connection, workspace_port=workspace)
    result = resolver.resolve(
        tenant_id=_TENANT,
        ingress=_ingress(activation_signal=ConversationActivationSignal.ORDINARY_MESSAGE),
    )
    assert result.workspace_id == _WORKSPACE


def test_mention_only_accepts_mention() -> None:
    repo = ConversationContextRepository(InMemoryDocumentStore())
    connection = _ConnectionPort()
    workspace = _WorkspacePort()
    assert repo.put_binding_if_absent(_binding(activation_policy=ConversationActivationPolicy.MENTION_ONLY)) is True
    _seed_workspace_context(repo, workspace, connection)
    resolver, _, _, _ = _resolver(repo, connection_port=connection, workspace_port=workspace)
    result = resolver.resolve(
        tenant_id=_TENANT,
        ingress=_ingress(activation_signal=ConversationActivationSignal.MENTION),
    )
    assert result.workspace_id == _WORKSPACE


def test_mention_only_accepts_thread_continuation() -> None:
    repo = ConversationContextRepository(InMemoryDocumentStore())
    connection = _ConnectionPort()
    workspace = _WorkspacePort()
    assert repo.put_binding_if_absent(_binding(activation_policy=ConversationActivationPolicy.MENTION_ONLY)) is True
    _seed_workspace_context(repo, workspace, connection)
    resolver, _, _, _ = _resolver(repo, connection_port=connection, workspace_port=workspace)
    result = resolver.resolve(
        tenant_id=_TENANT,
        ingress=_ingress(activation_signal=ConversationActivationSignal.THREAD_CONTINUATION),
    )
    assert result.workspace_id == _WORKSPACE


def test_mention_only_rejects_ordinary_message() -> None:
    repo = ConversationContextRepository(InMemoryDocumentStore())
    connection = _ConnectionPort(active_connections={(_TENANT, _CONNECTION)})
    workspace = _WorkspacePort()
    assert repo.put_binding_if_absent(_binding(activation_policy=ConversationActivationPolicy.MENTION_ONLY)) is True
    resolver, _, connection, workspace = _resolver(repo, connection_port=connection, workspace_port=workspace)
    with pytest.raises(ConversationContextResolutionError) as exc_info:
        resolver.resolve(
            tenant_id=_TENANT,
            ingress=_ingress(activation_signal=ConversationActivationSignal.ORDINARY_MESSAGE),
        )
    assert exc_info.value.error_code == "ACTIVATION_NOT_ALLOWED"
    assert workspace.calls == []


def test_explicit_command_accepts_only_explicit_command() -> None:
    repo = ConversationContextRepository(InMemoryDocumentStore())
    connection = _ConnectionPort()
    workspace = _WorkspacePort()
    assert repo.put_binding_if_absent(
        _binding(activation_policy=ConversationActivationPolicy.EXPLICIT_COMMAND)
    ) is True
    _seed_workspace_context(repo, workspace, connection)
    resolver, _, _, _ = _resolver(repo, connection_port=connection, workspace_port=workspace)
    result = resolver.resolve(
        tenant_id=_TENANT,
        ingress=_ingress(activation_signal=ConversationActivationSignal.EXPLICIT_COMMAND),
    )
    assert result.workspace_id == _WORKSPACE
    with pytest.raises(ConversationContextResolutionError) as exc_info:
        resolver.resolve(
            tenant_id=_TENANT,
            ingress=_ingress(activation_signal=ConversationActivationSignal.MENTION),
        )
    assert exc_info.value.error_code == "ACTIVATION_NOT_ALLOWED"


def test_personal_selection_resolves_exact_binding_principal_state() -> None:
    store = InMemoryDocumentStore()
    repo = _TrackingRepository(store)
    connection = _ConnectionPort(active_connections={(_TENANT, _CONNECTION)})
    workspace = _WorkspacePort(
        active_workspaces={(_TENANT, _WORKSPACE)},
        authorized={(_TENANT, _WORKSPACE, _PRINCIPAL)},
    )
    assert (
        repo.put_binding_if_absent(
            _binding(
                workspace_resolution_policy=ConversationWorkspaceResolutionPolicy.PERSONAL_SELECTION,
                workspace_id=None,
            )
        )
        is True
    )
    assert (
        repo.put_personal_state_if_absent(
            PersonalConversationStateV1(
                tenant_id=_TENANT,
                conversation_context_binding_id=_BINDING_ID,
                owner_principal_ref=_PRINCIPAL,
                selected_workspace_id=_WORKSPACE,
                configuration_version=1,
                updated_at=_NOW,
            )
        )
        is True
    )
    assert (
        repo.put_workspace_audience_policy_if_absent(
            WorkspaceConversationAudiencePolicyV1(
                tenant_id=_TENANT,
                workspace_id=_WORKSPACE,
                audience=WorkspaceConversationAudience.PERSONAL,
                configuration_version=1,
                updated_at=_NOW,
            )
        )
        is True
    )
    resolver, _, _, _ = _resolver(repo, connection_port=connection, workspace_port=workspace)
    result = resolver.resolve(tenant_id=_TENANT, ingress=_ingress())
    assert result.workspace_id == _WORKSPACE
    assert repo.personal_state_reads == 1


def test_missing_personal_state_has_no_global_fallback() -> None:
    repo = ConversationContextRepository(InMemoryDocumentStore())
    connection = _ConnectionPort(active_connections={(_TENANT, _CONNECTION)})
    workspace = _WorkspacePort()
    assert (
        repo.put_binding_if_absent(
            _binding(
                workspace_resolution_policy=ConversationWorkspaceResolutionPolicy.PERSONAL_SELECTION,
                workspace_id=None,
            )
        )
        is True
    )
    resolver, _, _, workspace = _resolver(repo, connection_port=connection, workspace_port=workspace)
    with pytest.raises(ConversationContextResolutionError) as exc_info:
        resolver.resolve(tenant_id=_TENANT, ingress=_ingress())
    assert exc_info.value.error_code == "PERSONAL_WORKSPACE_SELECTION_MISSING"
    assert workspace.calls == []


def test_fixed_workspace_never_reads_personal_state() -> None:
    store = InMemoryDocumentStore()
    repo = _TrackingRepository(store)
    connection = _ConnectionPort(active_connections={(_TENANT, _CONNECTION)})
    workspace = _WorkspacePort(
        active_workspaces={(_TENANT, _WORKSPACE)},
        authorized={(_TENANT, _WORKSPACE, _PRINCIPAL)},
    )
    _seed_personal_fixed(repo, workspace, connection)
    resolver, _, _, _ = _resolver(repo, connection_port=connection, workspace_port=workspace)
    result = resolver.resolve(tenant_id=_TENANT, ingress=_ingress())
    assert result.workspace_id == _WORKSPACE
    assert repo.personal_state_reads == 0


def test_shared_cannot_resolve_personal_workspace() -> None:
    repo = ConversationContextRepository(InMemoryDocumentStore())
    connection = _ConnectionPort(active_connections={(_TENANT, _CONNECTION)})
    workspace = _WorkspacePort(
        active_workspaces={(_TENANT, _WORKSPACE)},
        authorized={(_TENANT, _WORKSPACE, _PRINCIPAL)},
    )
    assert (
        repo.put_binding_if_absent(
            _binding(
                audience_mode=ConversationAudienceMode.SHARED,
                owner_principal_ref=None,
            )
        )
        is True
    )
    assert (
        repo.put_workspace_audience_policy_if_absent(
            WorkspaceConversationAudiencePolicyV1(
                tenant_id=_TENANT,
                workspace_id=_WORKSPACE,
                audience=WorkspaceConversationAudience.PERSONAL,
                configuration_version=1,
                updated_at=_NOW,
            )
        )
        is True
    )
    resolver, _, _, workspace = _resolver(repo, connection_port=connection, workspace_port=workspace)
    with pytest.raises(ConversationContextResolutionError) as exc_info:
        resolver.resolve(
            tenant_id=_TENANT,
            ingress=_ingress(observed_audience=ConversationObservedAudience.SHARED),
        )
    assert exc_info.value.error_code == "WORKSPACE_AUDIENCE_INCOMPATIBLE"
    assert workspace.calls == ["workspace_active"]


def test_personal_may_resolve_authorized_personal_workspace() -> None:
    repo = ConversationContextRepository(InMemoryDocumentStore())
    connection = _ConnectionPort(active_connections={(_TENANT, _CONNECTION)})
    workspace = _WorkspacePort(
        active_workspaces={(_TENANT, _WORKSPACE)},
        authorized={(_TENANT, _WORKSPACE, _PRINCIPAL)},
    )
    _seed_personal_fixed(repo, workspace, connection, audience=WorkspaceConversationAudience.PERSONAL)
    resolver, _, _, _ = _resolver(repo, connection_port=connection, workspace_port=workspace)
    result = resolver.resolve(tenant_id=_TENANT, ingress=_ingress())
    assert result.workspace_id == _WORKSPACE


def test_personal_may_resolve_authorized_shared_workspace() -> None:
    repo = ConversationContextRepository(InMemoryDocumentStore())
    connection = _ConnectionPort()
    workspace = _WorkspacePort()
    assert (
        repo.put_binding_if_absent(
            _binding(
                workspace_id=_WORKSPACE_SHARED,
            )
        )
        is True
    )
    _seed_workspace_context(
        repo,
        workspace,
        connection,
        workspace_id=_WORKSPACE_SHARED,
        audience=WorkspaceConversationAudience.SHARED,
    )
    resolver, _, _, _ = _resolver(repo, connection_port=connection, workspace_port=workspace)
    result = resolver.resolve(tenant_id=_TENANT, ingress=_ingress())
    assert result.workspace_id == _WORKSPACE_SHARED


def test_unauthorized_workspace_fails() -> None:
    repo = ConversationContextRepository(InMemoryDocumentStore())
    connection = _ConnectionPort(active_connections={(_TENANT, _CONNECTION)})
    workspace = _WorkspacePort(active_workspaces={(_TENANT, _WORKSPACE)})
    assert repo.put_binding_if_absent(_binding()) is True
    assert (
        repo.put_workspace_audience_policy_if_absent(
            WorkspaceConversationAudiencePolicyV1(
                tenant_id=_TENANT,
                workspace_id=_WORKSPACE,
                audience=WorkspaceConversationAudience.PERSONAL,
                configuration_version=1,
                updated_at=_NOW,
            )
        )
        is True
    )
    resolver, _, _, _ = _resolver(repo, connection_port=connection, workspace_port=workspace)
    with pytest.raises(ConversationContextResolutionError) as exc_info:
        resolver.resolve(tenant_id=_TENANT, ingress=_ingress())
    assert exc_info.value.error_code == "WORKSPACE_NOT_AUTHORIZED"


def test_exact_thread_reference_is_preserved() -> None:
    repo = ConversationContextRepository(InMemoryDocumentStore())
    connection = _ConnectionPort(active_connections={(_TENANT, _CONNECTION)})
    workspace = _WorkspacePort(
        active_workspaces={(_TENANT, _WORKSPACE)},
        authorized={(_TENANT, _WORKSPACE, _PRINCIPAL)},
    )
    _seed_personal_fixed(repo, workspace, connection)
    resolver, _, _, _ = _resolver(repo, connection_port=connection, workspace_port=workspace)
    result = resolver.resolve(
        tenant_id=_TENANT,
        ingress=_ingress(opaque_thread_ref="thread-exact-ref"),
    )
    assert result.canonical_thread_ref == "thread-exact-ref"


def test_two_tenants_cannot_resolve_each_others_binding_or_state() -> None:
    store = InMemoryDocumentStore()
    repo = ConversationContextRepository(store)
    assert repo.put_binding_if_absent(_binding()) is True
    assert (
        repo.put_personal_state_if_absent(
            PersonalConversationStateV1(
                tenant_id=_TENANT,
                conversation_context_binding_id=_BINDING_ID,
                owner_principal_ref=_PRINCIPAL,
                selected_workspace_id=_WORKSPACE,
                configuration_version=1,
                updated_at=_NOW,
            )
        )
        is True
    )
    connection = _ConnectionPort(active_connections={(_TENANT_B, _CONNECTION)})
    workspace = _WorkspacePort()
    resolver, _, _, _ = _resolver(repo, connection_port=connection, workspace_port=workspace)
    with pytest.raises(ConversationContextResolutionError) as exc_info:
        resolver.resolve(tenant_id=_TENANT_B, ingress=_ingress())
    assert exc_info.value.error_code == "NO_ACTIVE_BINDING"


def test_corrupt_binding_record_fails_closed_before_ports() -> None:
    store = InMemoryDocumentStore()
    partition_key = f"lkw.conversation_context:{_TENANT}:conversation_context_binding"
    corrupt = _binding(opaque_conversation_ref="conv.other")
    store.put(
        DocumentRecord(
            partition_key=partition_key,
            row_key=f"{_CONNECTION}\x1e{_CONVERSATION}\x1e{_BINDING_ID}",
            data=corrupt.model_dump(mode="json"),
        )
    )
    connection = _ConnectionPort(active_connections={(_TENANT, _CONNECTION)})
    workspace = _WorkspacePort(
        active_workspaces={(_TENANT, _WORKSPACE)},
        authorized={(_TENANT, _WORKSPACE, _PRINCIPAL)},
    )
    resolver, _, connection, workspace = _resolver(
        ConversationContextRepository(store),
        connection_port=connection,
        workspace_port=workspace,
    )
    with pytest.raises(ConversationContextRepositoryError) as exc_info:
        resolver.resolve(tenant_id=_TENANT, ingress=_ingress())
    assert exc_info.value.error_code == "conversation_context_record_identity_mismatch"
    assert connection.calls == []
    assert workspace.calls == []
