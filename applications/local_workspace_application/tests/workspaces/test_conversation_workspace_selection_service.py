# © Artur Czarnecki. All rights reserved.

"""Unit tests for durable personal conversation workspace selection."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.integrations.contracts.document_store import DocumentStore
from local_workspace_application.workspaces.conversation_context_models import (
    ConversationActivationPolicy,
    ConversationAudienceMode,
    ConversationExecutionContextV1,
    ConversationProductCapability,
    ConversationThreadContextPolicy,
    PersonalConversationStateV1,
)
from local_workspace_application.workspaces.conversation_context_repository import (
    ConversationContextRepository,
    ConversationContextRepositoryError,
)
from local_workspace_application.workspaces.conversation_workspace_selection_service import (
    ConversationWorkspaceSelectionError,
    ConversationWorkspaceSelectionResultV1,
    ConversationWorkspaceSelectionService,
)
from local_workspace_application.workspaces.service import ManagedWorkspaceService

pytestmark = pytest.mark.unit

_NOW = datetime(2024, 6, 1, 12, 0, tzinfo=UTC)
_UPDATED = datetime(2024, 6, 1, 12, 1, tzinfo=UTC)
_TENANT = "tenant-a"
_OTHER_TENANT = "tenant-b"
_BINDING = "binding-1"
_OTHER_BINDING = "binding-2"
_PRINCIPAL = "principal.alice"
_OTHER_PRINCIPAL = "principal.bob"
_WORKSPACE = "workspace-1"
_OTHER_WORKSPACE = "workspace-2"
_THREAD = "thread-1"


def _context(**overrides: object) -> ConversationExecutionContextV1:
    payload: dict[str, object] = {
        "tenant_id": _TENANT,
        "conversation_context_binding_id": _BINDING,
        "audience_mode": ConversationAudienceMode.PERSONAL,
        "workspace_id": _WORKSPACE,
        "principal_ref": _PRINCIPAL,
        "canonical_thread_ref": _THREAD,
        "activation_policy": ConversationActivationPolicy.ALWAYS,
        "thread_context_policy": ConversationThreadContextPolicy.CURRENT_THREAD_BOUNDED,
        "allowed_product_capabilities": frozenset(
            {ConversationProductCapability.WORKSPACE_SELECTION}
        ),
    }
    payload.update(overrides)
    return ConversationExecutionContextV1(**payload)  # type: ignore[arg-type]


def _workspace_service(
    *workspace_ids: str,
) -> Mock:
    service = Mock(spec=ManagedWorkspaceService)

    def get_workspace(*, tenant_id: str, workspace_id: str) -> object | None:
        if tenant_id == _TENANT and workspace_id in workspace_ids:
            return object()
        return None

    service.get_workspace.side_effect = get_workspace
    return service


def _service(
    repository: ConversationContextRepository,
    *,
    workspace_ids: tuple[str, ...] = (_WORKSPACE, _OTHER_WORKSPACE),
    clock: object = lambda: _NOW,
) -> tuple[ConversationWorkspaceSelectionService, Mock]:
    workspace_service = _workspace_service(*workspace_ids)
    return (
        ConversationWorkspaceSelectionService(
            repository,
            workspace_service,
            clock=clock,  # type: ignore[arg-type]
        ),
        workspace_service,
    )


def _state(
    *,
    tenant_id: str = _TENANT,
    binding_id: str = _BINDING,
    principal_ref: str = _PRINCIPAL,
    workspace_id: str = _WORKSPACE,
    version: int = 1,
    updated_at: datetime = _NOW,
) -> PersonalConversationStateV1:
    return PersonalConversationStateV1(
        tenant_id=tenant_id,
        conversation_context_binding_id=binding_id,
        owner_principal_ref=principal_ref,
        selected_workspace_id=workspace_id,
        configuration_version=version,
        updated_at=updated_at,
    )


def test_creates_selection_and_reconstructs_from_same_store() -> None:
    store = InMemoryDocumentStore()
    repository = ConversationContextRepository(store)
    service, _ = _service(repository)

    result = service.select_personal_workspace(
        execution_context=_context(),
        workspace_id=_WORKSPACE,
    )

    assert result.changed is True
    assert result.previous_workspace_id is None
    assert result.configuration_version == 1
    reconstructed = ConversationContextRepository(store).get_personal_state(
        tenant_id=_TENANT,
        conversation_context_binding_id=_BINDING,
        owner_principal_ref=_PRINCIPAL,
    )
    assert reconstructed == _state()


def test_updates_selection_with_incremented_version_and_previous_workspace() -> None:
    store = InMemoryDocumentStore()
    repository = ConversationContextRepository(store)
    clocks = iter((_NOW, _UPDATED))
    service, _ = _service(repository, clock=lambda: next(clocks))

    service.select_personal_workspace(
        execution_context=_context(),
        workspace_id=_WORKSPACE,
    )
    result = service.select_personal_workspace(
        execution_context=_context(),
        workspace_id=_OTHER_WORKSPACE,
    )

    assert result.changed is True
    assert result.previous_workspace_id == _WORKSPACE
    assert result.selected_workspace_id == _OTHER_WORKSPACE
    assert result.configuration_version == 2
    assert result.updated_at == _UPDATED
    loaded = ConversationContextRepository(store).get_personal_state(
        tenant_id=_TENANT,
        conversation_context_binding_id=_BINDING,
        owner_principal_ref=_PRINCIPAL,
    )
    assert loaded == _state(
        workspace_id=_OTHER_WORKSPACE,
        version=2,
        updated_at=_UPDATED,
    )


def test_idempotent_selection_does_not_replace_state() -> None:
    store = InMemoryDocumentStore()
    repository = Mock(
        wraps=ConversationContextRepository(store),
        spec=ConversationContextRepository,
    )
    service, _ = _service(repository)
    service.select_personal_workspace(
        execution_context=_context(),
        workspace_id=_WORKSPACE,
    )

    result = service.select_personal_workspace(
        execution_context=_context(),
        workspace_id=_WORKSPACE,
    )

    assert result.changed is False
    assert result.previous_workspace_id == _WORKSPACE
    assert result.configuration_version == 1
    assert result.updated_at == _NOW
    repository.replace_personal_state_if_match.assert_not_called()


def test_shared_context_is_rejected_before_repository_mutation() -> None:
    repository = Mock(spec=ConversationContextRepository)
    service, workspace_service = _service(repository)

    with pytest.raises(ConversationWorkspaceSelectionError) as exc_info:
        service.select_personal_workspace(
            execution_context=_context(
                audience_mode=ConversationAudienceMode.SHARED,
                allowed_product_capabilities=frozenset(
                    {ConversationProductCapability.READ_ONLY_ASK}
                ),
            ),
            workspace_id=_WORKSPACE,
        )

    assert exc_info.value.error_code == "workspace_selection_personal_context_required"
    workspace_service.get_workspace.assert_not_called()
    repository.get_personal_state.assert_not_called()


def test_missing_capability_is_rejected_even_when_target_is_current_workspace() -> None:
    repository = Mock(spec=ConversationContextRepository)
    service, workspace_service = _service(repository)

    with pytest.raises(ConversationWorkspaceSelectionError) as exc_info:
        service.select_personal_workspace(
            execution_context=_context(
                allowed_product_capabilities=frozenset(
                    {ConversationProductCapability.READ_ONLY_ASK}
                )
            ),
            workspace_id=_WORKSPACE,
        )

    assert exc_info.value.error_code == "workspace_selection_not_allowed"
    workspace_service.get_workspace.assert_not_called()
    repository.get_personal_state.assert_not_called()


@pytest.mark.parametrize("workspace_id", ["workspace-unknown", "workspace-cross-tenant"])
def test_unknown_or_cross_tenant_workspace_is_not_found(
    workspace_id: str,
) -> None:
    store = InMemoryDocumentStore()
    repository = ConversationContextRepository(store)
    service, _ = _service(repository)

    with pytest.raises(ConversationWorkspaceSelectionError) as exc_info:
        service.select_personal_workspace(
            execution_context=_context(),
            workspace_id=workspace_id,
        )

    assert exc_info.value.error_code == "workspace_not_found"
    assert repository.get_personal_state(
        tenant_id=_TENANT,
        conversation_context_binding_id=_BINDING,
        owner_principal_ref=_PRINCIPAL,
    ) is None


@pytest.mark.parametrize(
    ("binding_id", "principal_ref"),
    [
        (_BINDING, _OTHER_PRINCIPAL),
        (_OTHER_BINDING, _PRINCIPAL),
    ],
)
def test_existing_state_identity_mismatch_fails_closed(
    binding_id: str,
    principal_ref: str,
) -> None:
    repository = Mock(spec=ConversationContextRepository)
    repository.get_personal_state.return_value = _state(
        binding_id=binding_id,
        principal_ref=principal_ref,
    )
    service, _ = _service(repository)

    with pytest.raises(ConversationWorkspaceSelectionError) as exc_info:
        service.select_personal_workspace(
            execution_context=_context(),
            workspace_id=_OTHER_WORKSPACE,
        )

    assert exc_info.value.error_code == "conversation_context_identity_mismatch"
    repository.replace_personal_state_if_match.assert_not_called()


def test_initial_put_conflict_does_not_overwrite() -> None:
    repository = Mock(spec=ConversationContextRepository)
    repository.get_personal_state.return_value = None
    repository.put_personal_state_if_absent.return_value = False
    service, _ = _service(repository)

    with pytest.raises(ConversationWorkspaceSelectionError) as exc_info:
        service.select_personal_workspace(
            execution_context=_context(),
            workspace_id=_WORKSPACE,
        )

    assert exc_info.value.error_code == "workspace_selection_conflict"
    repository.put_personal_state_if_absent.assert_called_once()
    repository.replace_personal_state_if_match.assert_not_called()


def test_update_cas_conflict_preserves_current_winner_state() -> None:
    repository = Mock(spec=ConversationContextRepository)
    current = _state()
    repository.get_personal_state.return_value = current
    repository.replace_personal_state_if_match.return_value = False
    service, _ = _service(repository)

    with pytest.raises(ConversationWorkspaceSelectionError) as exc_info:
        service.select_personal_workspace(
            execution_context=_context(),
            workspace_id=_OTHER_WORKSPACE,
        )

    assert exc_info.value.error_code == "workspace_selection_conflict"
    assert repository.get_personal_state.return_value == current


def test_non_conditional_store_is_normalized_without_raw_error() -> None:
    repository = ConversationContextRepository(Mock(spec=DocumentStore))
    service, _ = _service(repository)

    with pytest.raises(ConversationWorkspaceSelectionError) as exc_info:
        service.select_personal_workspace(
            execution_context=_context(),
            workspace_id=_WORKSPACE,
        )

    assert exc_info.value.error_code == "conversation_context_storage_unavailable"
    assert str(exc_info.value) == "conversation_context_storage_unavailable"


@pytest.mark.parametrize(
    "repository_error",
    [
        "conversation_context_malformed_record",
        "conversation_context_record_identity_mismatch",
    ],
)
def test_known_repository_identity_errors_are_normalized(
    repository_error: str,
) -> None:
    repository = Mock(spec=ConversationContextRepository)
    repository.get_personal_state.side_effect = ConversationContextRepositoryError(
        repository_error
    )
    service, _ = _service(repository)

    with pytest.raises(ConversationWorkspaceSelectionError) as exc_info:
        service.select_personal_workspace(
            execution_context=_context(),
            workspace_id=_WORKSPACE,
        )

    assert exc_info.value.error_code == "conversation_context_identity_mismatch"
    assert str(exc_info.value) == "conversation_context_identity_mismatch"


def test_unknown_repository_error_is_storage_unavailable_and_safe() -> None:
    repository = Mock(spec=ConversationContextRepository)
    repository.get_personal_state.side_effect = RuntimeError(
        "tenant-a principal.alice binding-1"
    )
    service, _ = _service(repository)

    with pytest.raises(ConversationWorkspaceSelectionError) as exc_info:
        service.select_personal_workspace(
            execution_context=_context(),
            workspace_id=_WORKSPACE,
        )

    assert exc_info.value.error_code == "conversation_context_storage_unavailable"
    assert str(exc_info.value) == "conversation_context_storage_unavailable"
    assert "tenant-a" not in str(exc_info.value)


def test_invalid_clock_fails_before_initial_repository_write() -> None:
    repository = Mock(spec=ConversationContextRepository)
    repository.get_personal_state.return_value = None
    service, _ = _service(repository, clock=lambda: datetime(2024, 6, 1, 12, 0))

    with pytest.raises(ValueError, match="clock_must_produce_timezone_aware_utc"):
        service.select_personal_workspace(
            execution_context=_context(),
            workspace_id=_WORKSPACE,
        )

    repository.put_personal_state_if_absent.assert_not_called()


def test_result_rejects_non_utc_timestamp_and_exposes_only_safe_fields() -> None:
    assert set(ConversationWorkspaceSelectionResultV1.model_fields) == {
        "tenant_id",
        "conversation_context_binding_id",
        "owner_principal_ref",
        "selected_workspace_id",
        "previous_workspace_id",
        "configuration_version",
        "changed",
        "updated_at",
    }
    with pytest.raises(ValidationError, match="datetime_must_be_utc"):
        ConversationWorkspaceSelectionResultV1(
            tenant_id=_TENANT,
            conversation_context_binding_id=_BINDING,
            owner_principal_ref=_PRINCIPAL,
            selected_workspace_id=_WORKSPACE,
            previous_workspace_id=None,
            configuration_version=1,
            changed=True,
            updated_at=datetime(2024, 6, 1, 12, 0, tzinfo=timezone(timedelta(hours=2))),
        )
