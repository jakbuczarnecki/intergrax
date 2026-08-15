# © Artur Czarnecki. All rights reserved.

"""Atomic workspace deletion claim tests (LKW-PRODUCT-4B-R1)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from local_workspace_application.conversation.interaction_execution_models import (
    ConversationActionExecutionStatus,
    ConversationInteractionExecutionCommand,
)
from local_workspace_application.conversation.interaction_executor import (
    ConversationInteractionExecutor,
)
from local_workspace_application.conversation.interaction_models import (
    ConversationInteractionPlan,
    ConversationPlanningRequest,
    DestructiveActionConfirmPlannedAction,
)
from local_workspace_application.workspaces.conversation_context_models import (
    ConversationActivationPolicy,
    ConversationAudienceMode,
    ConversationExecutionContextV1,
    ConversationProductCapability,
    ConversationThreadContextPolicy,
)
from local_workspace_application.workspaces.destructive_action_confirmation import (
    DestructiveActionConfirmationV1,
    DestructiveActionKindV1,
    HmacDestructiveActionConfirmationCodec,
)
from local_workspace_application.workspaces.managed_files import ManagedFileCleanupPort
from local_workspace_application.workspaces.models import Workspace, WorkspaceStatus
from local_workspace_application.workspaces.repository import ManagedWorkspaceRepository
from local_workspace_application.workspaces.service import ManagedWorkspaceService
from local_workspace_application.workspaces.vector_cleanup import WorkspaceVectorCleanupPort
from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore

pytestmark = pytest.mark.unit

_NOW = datetime(2026, 8, 11, 10, 0, tzinfo=UTC)
_TENANT = "tenant-a"
_OTHER_TENANT = "tenant-b"
_WORKSPACE = "workspace-a"
_CONFIRM_SECRET = b"workspace-delete-claim-secret"
_CONFIRM_CODEC = HmacDestructiveActionConfirmationCodec(
    secret=_CONFIRM_SECRET,
    clock=lambda: _NOW,
)


class _TrackingVectorCleanup:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def delete_workspace_vectors(self, *, tenant_id: str, workspace_id: str) -> int:
        self.calls.append((tenant_id, workspace_id))
        return 0


class _FailingManagedFileCleanup:
    def delete_workspace_files(self, *, tenant_id: str, workspace_id: str) -> None:
        raise OSError("managed_file_cleanup_failed")


def _context() -> ConversationExecutionContextV1:
    return ConversationExecutionContextV1(
        tenant_id=_TENANT,
        conversation_context_binding_id="binding-1",
        workspace_id=_WORKSPACE,
        principal_ref="principal-1",
        canonical_thread_ref="thread-1",
        audience_mode=ConversationAudienceMode.PERSONAL,
        activation_policy=ConversationActivationPolicy.ALWAYS,
        thread_context_policy=ConversationThreadContextPolicy.CURRENT_THREAD_BOUNDED,
        allowed_product_capabilities=frozenset(ConversationProductCapability),
    )


def _workspace(*, revision: int = 1, status: WorkspaceStatus = WorkspaceStatus.ACTIVE) -> Workspace:
    return Workspace(
        workspace_id=_WORKSPACE,
        tenant_id=_TENANT,
        name="Alpha",
        status=status,
        workspace_revision=revision,
        created_at=_NOW,
        updated_at=_NOW,
    )


def _service(
    *,
    vector_cleanup: WorkspaceVectorCleanupPort | None = None,
    managed_file_cleanup: ManagedFileCleanupPort | None = None,
) -> tuple[ManagedWorkspaceService, ManagedWorkspaceRepository, _TrackingVectorCleanup | None]:
    store = InMemoryDocumentStore()
    repository = ManagedWorkspaceRepository(store)
    tracking = vector_cleanup if isinstance(vector_cleanup, _TrackingVectorCleanup) else None
    service = ManagedWorkspaceService(
        repository,
        vector_cleanup=vector_cleanup,
        managed_file_cleanup=managed_file_cleanup,
    )
    return service, repository, tracking


def _issue_token(
    *,
    revision: int = 1,
    tenant_id: str = _TENANT,
    workspace_id: str = _WORKSPACE,
    expires_at: datetime | None = None,
) -> str:
    return _CONFIRM_CODEC.issue(
        DestructiveActionConfirmationV1(
            token="",
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            action_kind=DestructiveActionKindV1.WORKSPACE_DELETE,
            target_id=workspace_id,
            expected_state_version=revision,
            expires_at=expires_at or (_NOW + timedelta(minutes=5)),
        )
    )


async def _confirm_delete(
    *,
    token: str,
    tenant_id: str = _TENANT,
    service: ManagedWorkspaceService,
    workspace_id: str = _WORKSPACE,
) -> ConversationActionExecutionStatus:
    executor = ConversationInteractionExecutor(
        workspace_service=service,  # type: ignore[arg-type]
        workspace_selection_service=SimpleNamespace(),  # type: ignore[arg-type]
        destructive_confirmation_codec=_CONFIRM_CODEC,
        clock=lambda: _NOW,
    )
    result = await executor.execute(
        ConversationInteractionExecutionCommand(
            tenant_id=tenant_id,
            planning_request=ConversationPlanningRequest(message_text="confirm delete"),
            interaction_plan=ConversationInteractionPlan(
                plan_version="2",
                actions=(
                    DestructiveActionConfirmPlannedAction(
                        action_id="confirm",
                        action_type="destructive.confirm",
                        confirmation_token=token,
                    ),
                ),
                response_mode="aggregate",
            ),
            execution_context=_context_for_workspace(workspace_id),
        )
    )
    return result.action_results[0].status


def _context_for_workspace(workspace_id: str) -> ConversationExecutionContextV1:
    return ConversationExecutionContextV1(
        tenant_id=_TENANT,
        conversation_context_binding_id="binding-1",
        workspace_id=workspace_id,
        principal_ref="principal-1",
        canonical_thread_ref="thread-1",
        audience_mode=ConversationAudienceMode.PERSONAL,
        activation_policy=ConversationActivationPolicy.ALWAYS,
        thread_context_policy=ConversationThreadContextPolicy.CURRENT_THREAD_BOUNDED,
        allowed_product_capabilities=frozenset(ConversationProductCapability),
    )


@pytest.mark.asyncio
async def test_valid_revision_claims_delete_and_runs_cleanup() -> None:
    tracking = _TrackingVectorCleanup()
    service, repository, _ = _service(vector_cleanup=tracking)
    created = service.create_workspace(tenant_id=_TENANT, name="Alpha")
    token = _issue_token(revision=created.workspace_revision, workspace_id=created.workspace_id)

    status = await _confirm_delete(
        token=token,
        service=service,
        workspace_id=created.workspace_id,
    )

    assert status is ConversationActionExecutionStatus.COMPLETED
    assert tracking.calls == [(_TENANT, created.workspace_id)]
    assert repository.get_workspace(tenant_id=_TENANT, workspace_id=created.workspace_id) is None


@pytest.mark.asyncio
async def test_stale_revision_blocks_delete_without_cleanup() -> None:
    tracking = _TrackingVectorCleanup()
    service, repository, _ = _service(vector_cleanup=tracking)
    created = service.create_workspace(tenant_id=_TENANT, name="Alpha")
    bumped = created.model_copy(
        update={"workspace_revision": 2, "updated_at": _NOW + timedelta(seconds=1)}
    )
    assert service.replace_workspace_if_match(created, bumped)
    token = _issue_token(revision=1, workspace_id=created.workspace_id)

    status = await _confirm_delete(
        token=token,
        service=service,
        workspace_id=created.workspace_id,
    )

    assert status is ConversationActionExecutionStatus.FAILED
    assert tracking.calls == []
    reloaded = repository.get_workspace(tenant_id=_TENANT, workspace_id=created.workspace_id)
    assert reloaded is not None
    assert reloaded.status is WorkspaceStatus.ACTIVE


@pytest.mark.asyncio
async def test_concurrent_update_before_claim_fails_stale_without_cleanup() -> None:
    tracking = _TrackingVectorCleanup()
    service, repository, _ = _service(vector_cleanup=tracking)
    created = service.create_workspace(tenant_id=_TENANT, name="Alpha")
    token = _issue_token(revision=created.workspace_revision, workspace_id=created.workspace_id)

    current = repository.get_workspace(tenant_id=_TENANT, workspace_id=created.workspace_id)
    assert current is not None
    bumped = current.model_copy(
        update={"workspace_revision": 2, "updated_at": _NOW + timedelta(seconds=1)}
    )
    assert service.replace_workspace_if_match(current, bumped)

    status = await _confirm_delete(
        token=token,
        service=service,
        workspace_id=created.workspace_id,
    )

    assert status is ConversationActionExecutionStatus.FAILED
    assert tracking.calls == []


def test_claim_wins_then_ordinary_workspace_update_cannot_succeed() -> None:
    service, repository, _ = _service()
    created = service.create_workspace(tenant_id=_TENANT, name="Alpha")
    current = repository.get_workspace(tenant_id=_TENANT, workspace_id=created.workspace_id)
    assert current is not None

    outcome, _ = service.delete_workspace_with_revision_claim(
        tenant_id=_TENANT,
        workspace_id=created.workspace_id,
        expected_revision=current.workspace_revision,
    )
    assert outcome == "deleted"

    claimed = repository.get_workspace(tenant_id=_TENANT, workspace_id=created.workspace_id)
    assert claimed is None


def test_claim_blocks_subsequent_active_revision_update() -> None:
    service, repository, _ = _service()
    created = service.create_workspace(tenant_id=_TENANT, name="Alpha")
    current = repository.get_workspace(tenant_id=_TENANT, workspace_id=created.workspace_id)
    assert current is not None
    assert repository.claim_workspace_deletion_if_match(current, claimed_at=_NOW)

    deleting = repository.get_workspace(tenant_id=_TENANT, workspace_id=created.workspace_id)
    assert deleting is not None
    assert deleting.status is WorkspaceStatus.DELETING
    assert deleting.workspace_revision == 2

    stale_update = current.model_copy(
        update={"name": "Renamed", "workspace_revision": 3, "updated_at": _NOW + timedelta(seconds=1)}
    )
    assert service.replace_workspace_if_match(current, stale_update) is False


def test_deleting_workspace_sealed_against_current_record_mutation() -> None:
    service, repository, _ = _service()
    created = service.create_workspace(tenant_id=_TENANT, name="Alpha")
    current = repository.get_workspace(tenant_id=_TENANT, workspace_id=created.workspace_id)
    assert current is not None
    assert repository.claim_workspace_deletion_if_match(current, claimed_at=_NOW)

    deleting = repository.get_workspace(tenant_id=_TENANT, workspace_id=created.workspace_id)
    assert deleting is not None
    renamed = deleting.model_copy(
        update={"name": "Renamed", "workspace_revision": 3, "updated_at": _NOW + timedelta(seconds=1)}
    )
    assert service.replace_workspace_if_match(deleting, renamed) is False

    reloaded = repository.get_workspace(tenant_id=_TENANT, workspace_id=created.workspace_id)
    assert reloaded is not None
    assert reloaded.status is WorkspaceStatus.DELETING
    assert reloaded.name == "Alpha"
    assert reloaded.workspace_revision == 2


def test_deleting_cannot_reactivate_to_active() -> None:
    service, repository, _ = _service()
    created = service.create_workspace(tenant_id=_TENANT, name="Alpha")
    current = repository.get_workspace(tenant_id=_TENANT, workspace_id=created.workspace_id)
    assert current is not None
    assert repository.claim_workspace_deletion_if_match(current, claimed_at=_NOW)

    deleting = repository.get_workspace(tenant_id=_TENANT, workspace_id=created.workspace_id)
    assert deleting is not None
    reactivated = deleting.model_copy(
        update={
            "status": WorkspaceStatus.ACTIVE,
            "workspace_revision": 3,
            "updated_at": _NOW + timedelta(seconds=1),
        }
    )
    assert service.replace_workspace_if_match(deleting, reactivated) is False

    reloaded = repository.get_workspace(tenant_id=_TENANT, workspace_id=created.workspace_id)
    assert reloaded is not None
    assert reloaded.status is WorkspaceStatus.DELETING


def test_normal_active_mutation_succeeds_with_monotonic_revision() -> None:
    service, repository, _ = _service()
    created = service.create_workspace(tenant_id=_TENANT, name="Alpha")
    bumped = created.model_copy(
        update={"workspace_revision": 2, "updated_at": _NOW + timedelta(seconds=1)}
    )
    assert service.replace_workspace_if_match(created, bumped)

    reloaded = repository.get_workspace(tenant_id=_TENANT, workspace_id=created.workspace_id)
    assert reloaded is not None
    assert reloaded.workspace_revision == 2
    assert reloaded.status is WorkspaceStatus.ACTIVE


def test_ordinary_mutation_rejects_invalid_revision_progression() -> None:
    service, repository, _ = _service()
    created = service.create_workspace(tenant_id=_TENANT, name="Alpha")

    skipped = created.model_copy(
        update={"workspace_revision": 3, "updated_at": _NOW + timedelta(seconds=1)}
    )
    assert service.replace_workspace_if_match(created, skipped) is False

    same_revision = created.model_copy(
        update={"name": "Renamed", "updated_at": _NOW + timedelta(seconds=1)}
    )
    assert service.replace_workspace_if_match(created, same_revision) is False

    reloaded = repository.get_workspace(tenant_id=_TENANT, workspace_id=created.workspace_id)
    assert reloaded is not None
    assert reloaded.workspace_revision == 1
    assert reloaded.name == "Alpha"


def test_deletion_claim_still_transitions_active_to_deleting() -> None:
    service, repository, _ = _service()
    created = service.create_workspace(tenant_id=_TENANT, name="Alpha")
    current = repository.get_workspace(tenant_id=_TENANT, workspace_id=created.workspace_id)
    assert current is not None

    assert repository.claim_workspace_deletion_if_match(current, claimed_at=_NOW)

    deleting = repository.get_workspace(tenant_id=_TENANT, workspace_id=created.workspace_id)
    assert deleting is not None
    assert deleting.status is WorkspaceStatus.DELETING
    assert deleting.workspace_revision == current.workspace_revision + 1


def test_deletion_resume_and_finalize_after_claim() -> None:
    tracking = _TrackingVectorCleanup()
    service, repository, _ = _service(
        vector_cleanup=tracking,
        managed_file_cleanup=_FailingManagedFileCleanup(),
    )
    created = service.create_workspace(tenant_id=_TENANT, name="Alpha")

    with pytest.raises(OSError, match="managed_file_cleanup_failed"):
        service.delete_workspace_with_revision_claim(
            tenant_id=_TENANT,
            workspace_id=created.workspace_id,
            expected_revision=created.workspace_revision,
        )

    service._managed_file_cleanup = None  # type: ignore[method-assign]
    outcome, name = service.delete_workspace_with_revision_claim(
        tenant_id=_TENANT,
        workspace_id=created.workspace_id,
        expected_revision=created.workspace_revision,
    )

    assert outcome == "deleted"
    assert name == "Alpha"
    assert len(tracking.calls) == 2
    assert repository.get_workspace(tenant_id=_TENANT, workspace_id=created.workspace_id) is None


@pytest.mark.asyncio
async def test_cross_tenant_token_fails_without_cleanup() -> None:
    tracking = _TrackingVectorCleanup()
    service, _, _ = _service(vector_cleanup=tracking)
    created = service.create_workspace(tenant_id=_TENANT, name="Alpha")
    token = _issue_token(revision=created.workspace_revision, tenant_id=_OTHER_TENANT)

    status = await _confirm_delete(
        token=token,
        service=service,
        workspace_id=created.workspace_id,
    )

    assert status is ConversationActionExecutionStatus.FAILED
    assert tracking.calls == []


@pytest.mark.asyncio
async def test_expired_token_fails_without_cleanup() -> None:
    tracking = _TrackingVectorCleanup()
    service, _, _ = _service(vector_cleanup=tracking)
    created = service.create_workspace(tenant_id=_TENANT, name="Alpha")
    token = _issue_token(
        revision=created.workspace_revision,
        workspace_id=created.workspace_id,
        expires_at=_NOW - timedelta(minutes=1),
    )

    status = await _confirm_delete(
        token=token,
        service=service,
        workspace_id=created.workspace_id,
    )

    assert status is ConversationActionExecutionStatus.FAILED
    assert tracking.calls == []


@pytest.mark.asyncio
async def test_tampered_token_fails_without_cleanup() -> None:
    tracking = _TrackingVectorCleanup()
    service, _, _ = _service(vector_cleanup=tracking)
    created = service.create_workspace(tenant_id=_TENANT, name="Alpha")
    token = _issue_token(revision=created.workspace_revision, workspace_id=created.workspace_id) + "tampered"

    status = await _confirm_delete(
        token=token,
        service=service,
        workspace_id=created.workspace_id,
    )

    assert status is ConversationActionExecutionStatus.FAILED
    assert tracking.calls == []


@pytest.mark.asyncio
async def test_replay_after_successful_delete_is_safe() -> None:
    tracking = _TrackingVectorCleanup()
    service, repository, _ = _service(vector_cleanup=tracking)
    created = service.create_workspace(tenant_id=_TENANT, name="Alpha")
    token = _issue_token(revision=created.workspace_revision, workspace_id=created.workspace_id)

    first = await _confirm_delete(
        token=token,
        service=service,
        workspace_id=created.workspace_id,
    )
    second = await _confirm_delete(
        token=token,
        service=service,
        workspace_id=created.workspace_id,
    )

    assert first is ConversationActionExecutionStatus.COMPLETED
    assert second is ConversationActionExecutionStatus.FAILED
    assert len(tracking.calls) == 1
    assert repository.get_workspace(tenant_id=_TENANT, workspace_id=created.workspace_id) is None


def test_cleanup_failure_after_claim_leaves_deleting_state() -> None:
    service, repository, _ = _service(managed_file_cleanup=_FailingManagedFileCleanup())
    created = service.create_workspace(tenant_id=_TENANT, name="Alpha")

    with pytest.raises(OSError, match="managed_file_cleanup_failed"):
        service.delete_workspace_with_revision_claim(
            tenant_id=_TENANT,
            workspace_id=created.workspace_id,
            expected_revision=created.workspace_revision,
        )

    reloaded = repository.get_workspace(tenant_id=_TENANT, workspace_id=created.workspace_id)
    assert reloaded is not None
    assert reloaded.status is WorkspaceStatus.DELETING
    assert reloaded.workspace_revision == 2


@pytest.mark.asyncio
async def test_replay_after_claimed_cleanup_failure_resumes_safely() -> None:
    tracking = _TrackingVectorCleanup()
    service, repository, _ = _service(
        vector_cleanup=tracking,
        managed_file_cleanup=_FailingManagedFileCleanup(),
    )
    created = service.create_workspace(tenant_id=_TENANT, name="Alpha")
    token = _issue_token(revision=created.workspace_revision, workspace_id=created.workspace_id)

    first = await _confirm_delete(
        token=token,
        service=service,
        workspace_id=created.workspace_id,
    )
    assert first is ConversationActionExecutionStatus.FAILED

    service._managed_file_cleanup = None  # type: ignore[method-assign]
    second = await _confirm_delete(
        token=token,
        service=service,
        workspace_id=created.workspace_id,
    )

    assert second is ConversationActionExecutionStatus.COMPLETED
    assert len(tracking.calls) == 2
    assert repository.get_workspace(tenant_id=_TENANT, workspace_id=created.workspace_id) is None
