# © Artur Czarnecki. All rights reserved.

"""PRODUCT-4B daily knowledge lifecycle conversational path tests."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from local_workspace_application.conversation.conversation_setup_onboarding import (
    ConversationSetupOnboardingPresenter,
)
from local_workspace_application.conversation.interaction_execution_models import (
    ConversationActionExecutionStatus,
    ConversationInteractionExecutionCommand,
    ConversationInteractionOverallStatus,
)
from local_workspace_application.conversation.interaction_executor import (
    ConversationInteractionExecutor,
)
from local_workspace_application.conversation.interaction_models import (
    ConversationInteractionPlan,
    ConversationPlanningRequest,
    ConversationPlanningWorkspace,
    DestructiveActionConfirmPlannedAction,
    KnowledgeInventoryFilter,
    KnowledgeInventoryListPlannedAction,
    KnowledgeOperationExecutePlannedAction,
    KnowledgeOperationKind,
    KnowledgeTargetReferenceKind,
    WorkspaceDeletePlannedAction,
    WorkspaceListPlannedAction,
    WorkspaceReference,
    WorkspaceReferenceKind,
)
from local_workspace_application.conversation.interaction_response_renderer import (
    ConversationInteractionResponseRenderer,
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
from local_workspace_application.workspaces.knowledge_inspection_operations_service import (
    KnowledgeAccessModeV1,
    KnowledgeInventoryItemV1,
    KnowledgeInventorySummaryV1,
    KnowledgeInventoryV1,
    KnowledgeOperationError,
    KnowledgeOperationResultV1,
    KnowledgeOperationV1,
    KnowledgeRevisionKindV1,
)
from local_workspace_application.workspaces.models import Workspace, WorkspaceStatus
from local_workspace_application.workspaces.repository import ManagedWorkspaceRepository
from local_workspace_application.workspaces.service import ManagedWorkspaceService
from local_workspace_application.workspaces.workspace_setup_snapshot_service import (
    SetupKnowledgeSummaryV1,
    SetupNextActionV1,
    SetupPhaseV1,
    WorkspaceSetupSnapshotV1,
)
from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore

pytestmark = pytest.mark.unit

_NOW = datetime(2026, 8, 11, 10, 0, tzinfo=UTC)
_TENANT = "tenant-a"
_WORKSPACE = "workspace-a"
_CONFIRM_SECRET = b"product-4b-confirm-secret"
_CONFIRM_CODEC = HmacDestructiveActionConfirmationCodec(
    secret=_CONFIRM_SECRET,
    clock=lambda: _NOW,
)


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


def _reference(kind: WorkspaceReferenceKind, value: str | None = None) -> WorkspaceReference:
    return WorkspaceReference(kind=kind, value=value)


def _item(
    *,
    item_id: str = "indexed:binding-1",
    label: str = "Project Drive",
    mode: KnowledgeAccessModeV1 = KnowledgeAccessModeV1.INDEXED,
    state: str = "active",
    revision: int = 3,
    runtime_available: bool | None = None,
    last_sync: datetime | None = _NOW,
    actions: tuple[KnowledgeOperationV1, ...] = (
        KnowledgeOperationV1.SYNC,
        KnowledgeOperationV1.DISABLE,
        KnowledgeOperationV1.DETACH,
    ),
) -> KnowledgeInventoryItemV1:
    return KnowledgeInventoryItemV1(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        knowledge_item_id=item_id,
        mode=mode,
        display_label=label,
        lifecycle_state=state,
        enabled=state != "disabled",
        detached=False,
        runtime_available=runtime_available,
        sync_state="ready" if state == "active" else None,
        last_successful_sync_at=last_sync,
        revision=revision,
        revision_kind=KnowledgeRevisionKindV1.LIFECYCLE,
        available_actions=actions,
        updated_at=_NOW,
    )


class InspectionFake:
    def __init__(self, items: tuple[KnowledgeInventoryItemV1, ...]) -> None:
        self._items = items

    def list_items(self, *, tenant_id: str, workspace_id: str) -> KnowledgeInventoryV1:
        return KnowledgeInventoryV1(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            items=self._items,
            summary=KnowledgeInventorySummaryV1(
                total=len(self._items),
                indexed=sum(i.mode is KnowledgeAccessModeV1.INDEXED for i in self._items),
                live=sum(i.mode is KnowledgeAccessModeV1.LIVE for i in self._items),
                active=sum(i.lifecycle_state == "active" for i in self._items),
                disabled=sum(i.lifecycle_state == "disabled" for i in self._items),
                attention_required=0,
            ),
            updated_at=_NOW,
        )

    def get_item(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        knowledge_item_id: str,
    ) -> KnowledgeInventoryItemV1:
        for item in self._items:
            if item.knowledge_item_id == knowledge_item_id:
                return item
        raise RuntimeError("knowledge_item_not_found")


class OperationsFake:
    def __init__(self) -> None:
        self.calls: list[KnowledgeOperationV1] = []

    async def execute(self, command) -> KnowledgeOperationResultV1:
        self.calls.append(command.operation)
        item = _item(revision=command.expected_revision + 1)
        return KnowledgeOperationResultV1(item=item, operation=command.operation)


class WorkspaceServiceFake:
    def __init__(self, workspace: Workspace) -> None:
        self._workspace = workspace
        self.deleted = False

    def list_workspaces(self, *, tenant_id: str) -> list[Workspace]:
        return [self._workspace]

    def get_workspace(self, *, tenant_id: str, workspace_id: str) -> Workspace | None:
        if tenant_id != _TENANT or workspace_id != self._workspace.workspace_id:
            return None
        return self._workspace

    def delete_workspace(self, *, tenant_id: str, workspace_id: str) -> bool:
        if tenant_id != _TENANT or workspace_id != self._workspace.workspace_id:
            return False
        self.deleted = True
        return True


def _executor(
    *,
    workspace: Workspace,
    inspection: InspectionFake,
    operations: OperationsFake,
) -> ConversationInteractionExecutor:
    return ConversationInteractionExecutor(
        workspace_service=WorkspaceServiceFake(workspace),  # type: ignore[arg-type]
        workspace_selection_service=SimpleNamespace(),  # type: ignore[arg-type]
        knowledge_inspection_service=inspection,  # type: ignore[arg-type]
        knowledge_operations_service=operations,  # type: ignore[arg-type]
        destructive_confirmation_codec=_CONFIRM_CODEC,
        clock=lambda: _NOW,
        execution_id_factory=lambda: "exec-1",
    )


@pytest.mark.asyncio
async def test_inventory_list_renders_labels_and_attention() -> None:
    items = (
        _item(label="Project Drive"),
        _item(
            item_id="live:binding-2",
            label="Slack Support",
            mode=KnowledgeAccessModeV1.LIVE,
            runtime_available=False,
            last_sync=None,
            state="active",
            actions=(KnowledgeOperationV1.DISABLE, KnowledgeOperationV1.DETACH),
        ),
    )
    workspace = Workspace(
        workspace_id=_WORKSPACE,
        tenant_id=_TENANT,
        name="Alpha",
        status=WorkspaceStatus.ACTIVE,
        workspace_revision=1,
        created_at=_NOW,
        updated_at=_NOW,
    )
    executor = _executor(
        workspace=workspace,
        inspection=InspectionFake(items),
        operations=OperationsFake(),
    )
    plan = ConversationInteractionPlan(
        plan_version="2",
        actions=(
            WorkspaceListPlannedAction(action_id="list-ws", action_type="workspace.list"),
            KnowledgeInventoryListPlannedAction(
                action_id="inventory",
                action_type="knowledge.inventory.list",
                workspace=_reference(WorkspaceReferenceKind.active),
                inventory_filter=KnowledgeInventoryFilter.all,
            ),
        ),
        response_mode="aggregate",
    )
    result = await executor.execute(
        ConversationInteractionExecutionCommand(
            tenant_id=_TENANT,
            planning_request=ConversationPlanningRequest(
                message_text="pokaż źródła",
                available_workspaces=(
                    ConversationPlanningWorkspace(
                        workspace_id=_WORKSPACE,
                        name="Alpha",
                        is_active=True,
                    ),
                ),
                active_workspace_id=_WORKSPACE,
            ),
            interaction_plan=plan,
            execution_context=_context(),
        )
    )
    renderer = ConversationInteractionResponseRenderer()
    text = renderer.render(result)
    assert "1. Alpha (active)" in text
    assert "Project Drive" in text
    assert "Slack Support" in text
    assert "unavailable" in text


@pytest.mark.asyncio
async def test_sync_and_disable_operations_execute() -> None:
    items = (_item(),)
    workspace = Workspace(
        workspace_id=_WORKSPACE,
        tenant_id=_TENANT,
        name="Alpha",
        status=WorkspaceStatus.ACTIVE,
        workspace_revision=1,
        created_at=_NOW,
        updated_at=_NOW,
    )
    operations = OperationsFake()
    executor = _executor(
        workspace=workspace,
        inspection=InspectionFake(items),
        operations=operations,
    )
    plan = ConversationInteractionPlan(
        plan_version="2",
        actions=(
            KnowledgeOperationExecutePlannedAction(
                action_id="sync",
                action_type="knowledge.operation.execute",
                workspace=_reference(WorkspaceReferenceKind.active),
                operation=KnowledgeOperationKind.sync,
                target_reference_kind=KnowledgeTargetReferenceKind.ordinal,
                target_reference="1",
            ),
            KnowledgeOperationExecutePlannedAction(
                action_id="disable",
                action_type="knowledge.operation.execute",
                workspace=_reference(WorkspaceReferenceKind.active),
                operation=KnowledgeOperationKind.disable,
                target_reference_kind=KnowledgeTargetReferenceKind.display_label,
                target_reference="Project Drive",
            ),
        ),
        response_mode="aggregate",
    )
    result = await executor.execute(
        ConversationInteractionExecutionCommand(
            tenant_id=_TENANT,
            planning_request=ConversationPlanningRequest(message_text="sync and disable"),
            interaction_plan=plan,
            execution_context=_context(),
        )
    )
    assert result.status is ConversationInteractionOverallStatus.COMPLETED
    assert operations.calls == [
        KnowledgeOperationV1.SYNC,
        KnowledgeOperationV1.DISABLE,
    ]


@pytest.mark.asyncio
async def test_detach_requires_confirmation_then_executes() -> None:
    items = (_item(),)
    workspace = Workspace(
        workspace_id=_WORKSPACE,
        tenant_id=_TENANT,
        name="Alpha",
        status=WorkspaceStatus.ACTIVE,
        workspace_revision=1,
        created_at=_NOW,
        updated_at=_NOW,
    )
    operations = OperationsFake()
    executor = _executor(
        workspace=workspace,
        inspection=InspectionFake(items),
        operations=operations,
    )
    detach_plan = ConversationInteractionPlan(
        plan_version="2",
        actions=(
            KnowledgeOperationExecutePlannedAction(
                action_id="detach",
                action_type="knowledge.operation.execute",
                workspace=_reference(WorkspaceReferenceKind.active),
                operation=KnowledgeOperationKind.detach,
                target_reference_kind=KnowledgeTargetReferenceKind.ordinal,
                target_reference="1",
            ),
        ),
        response_mode="aggregate",
    )
    first = await executor.execute(
        ConversationInteractionExecutionCommand(
            tenant_id=_TENANT,
            planning_request=ConversationPlanningRequest(message_text="detach source 1"),
            interaction_plan=detach_plan,
            execution_context=_context(),
        )
    )
    token = first.action_results[0].artifact.data["confirmation_token"]
    confirm_plan = ConversationInteractionPlan(
        plan_version="2",
        actions=(
            DestructiveActionConfirmPlannedAction(
                action_id="confirm",
                action_type="destructive.confirm",
                confirmation_token=token,
            ),
        ),
        response_mode="aggregate",
    )
    second = await executor.execute(
        ConversationInteractionExecutionCommand(
            tenant_id=_TENANT,
            planning_request=ConversationPlanningRequest(message_text="yes confirm"),
            interaction_plan=confirm_plan,
            execution_context=_context(),
        )
    )
    assert second.action_results[0].status is ConversationActionExecutionStatus.COMPLETED, (
        second.action_results[0].error
    )
    assert operations.calls == [KnowledgeOperationV1.DETACH]
    renderer = ConversationInteractionResponseRenderer()
    assert "irreversible" in renderer.render(first)
    assert "detached" in renderer.render(second).casefold()


@pytest.mark.asyncio
async def test_invalid_detach_confirmation_fails_closed() -> None:
    items = (_item(),)
    workspace = Workspace(
        workspace_id=_WORKSPACE,
        tenant_id=_TENANT,
        name="Alpha",
        status=WorkspaceStatus.ACTIVE,
        workspace_revision=1,
        created_at=_NOW,
        updated_at=_NOW,
    )
    executor = _executor(
        workspace=workspace,
        inspection=InspectionFake(items),
        operations=OperationsFake(),
    )
    plan = ConversationInteractionPlan(
        plan_version="2",
        actions=(
            KnowledgeOperationExecutePlannedAction(
                action_id="detach",
                action_type="knowledge.operation.execute",
                workspace=_reference(WorkspaceReferenceKind.active),
                operation=KnowledgeOperationKind.detach,
                target_reference_kind=KnowledgeTargetReferenceKind.ordinal,
                target_reference="1",
                confirmation_token="not-a-valid-token",
            ),
        ),
        response_mode="aggregate",
    )
    result = await executor.execute(
        ConversationInteractionExecutionCommand(
            tenant_id=_TENANT,
            planning_request=ConversationPlanningRequest(message_text="confirm detach"),
            interaction_plan=plan,
            execution_context=_context(),
        )
    )
    assert result.action_results[0].status is ConversationActionExecutionStatus.FAILED


@pytest.mark.asyncio
async def test_workspace_delete_requires_confirmation() -> None:
    workspace = Workspace(
        workspace_id=_WORKSPACE,
        tenant_id=_TENANT,
        name="Alpha",
        status=WorkspaceStatus.ACTIVE,
        workspace_revision=2,
        created_at=_NOW,
        updated_at=_NOW,
    )
    service = WorkspaceServiceFake(workspace)
    executor = ConversationInteractionExecutor(
        workspace_service=service,  # type: ignore[arg-type]
        workspace_selection_service=SimpleNamespace(),  # type: ignore[arg-type]
        destructive_confirmation_codec=_CONFIRM_CODEC,
        clock=lambda: _NOW,
    )
    plan = ConversationInteractionPlan(
        plan_version="2",
        actions=(
            WorkspaceDeletePlannedAction(
                action_id="delete",
                action_type="workspace.delete",
                workspace=_reference(WorkspaceReferenceKind.name, "Alpha"),
            ),
        ),
        response_mode="aggregate",
    )
    result = await executor.execute(
        ConversationInteractionExecutionCommand(
            tenant_id=_TENANT,
            planning_request=ConversationPlanningRequest(
                message_text="usuń workspace Alpha",
                available_workspaces=(
                    ConversationPlanningWorkspace(
                        workspace_id=_WORKSPACE,
                        name="Alpha",
                        is_active=True,
                    ),
                ),
                active_workspace_id=_WORKSPACE,
            ),
            interaction_plan=plan,
            execution_context=_context(),
        )
    )
    assert service.deleted is False
    token = result.action_results[0].artifact.data["confirmation_token"]
    confirm = await executor.execute(
        ConversationInteractionExecutionCommand(
            tenant_id=_TENANT,
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
            execution_context=_context(),
        )
    )
    assert confirm.action_results[0].status is ConversationActionExecutionStatus.COMPLETED, (
        confirm.action_results[0].error
    )
    assert service.deleted is True
    renderer = ConversationInteractionResponseRenderer()
    assert "irreversible" in renderer.render(result)
    assert "deleted" in renderer.render(confirm).casefold()


@pytest.mark.asyncio
async def test_stale_workspace_revision_blocks_delete_confirmation() -> None:
    workspace = Workspace(
        workspace_id=_WORKSPACE,
        tenant_id=_TENANT,
        name="Alpha",
        status=WorkspaceStatus.ACTIVE,
        workspace_revision=2,
        created_at=_NOW,
        updated_at=_NOW,
    )
    service = WorkspaceServiceFake(workspace)
    token = _CONFIRM_CODEC.issue(
        DestructiveActionConfirmationV1(
            token="",
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            action_kind=DestructiveActionKindV1.WORKSPACE_DELETE,
            target_id=_WORKSPACE,
            expected_state_version=1,
            expires_at=_NOW + timedelta(minutes=5),
        )
    )
    executor = ConversationInteractionExecutor(
        workspace_service=service,  # type: ignore[arg-type]
        workspace_selection_service=SimpleNamespace(),  # type: ignore[arg-type]
        destructive_confirmation_codec=_CONFIRM_CODEC,
        clock=lambda: _NOW,
    )
    result = await executor.execute(
        ConversationInteractionExecutionCommand(
            tenant_id=_TENANT,
            planning_request=ConversationPlanningRequest(message_text="confirm"),
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
            execution_context=_context(),
        )
    )
    assert result.action_results[0].status is ConversationActionExecutionStatus.FAILED
    assert service.deleted is False


def test_workspace_revision_cas_and_persistence() -> None:
    store = InMemoryDocumentStore()
    repository = ManagedWorkspaceRepository(store)
    service = ManagedWorkspaceService(repository)
    created = service.create_workspace(tenant_id=_TENANT, name="Alpha")
    assert created.workspace_revision == 1
    loaded = repository.get_workspace(tenant_id=_TENANT, workspace_id=created.workspace_id)
    assert loaded is not None
    assert loaded.workspace_revision == 1
    bumped = created.model_copy(
        update={"workspace_revision": 2, "updated_at": _NOW + timedelta(seconds=1)}
    )
    assert service.replace_workspace_if_match(created, bumped)
    reloaded = repository.get_workspace(
        tenant_id=_TENANT,
        workspace_id=created.workspace_id,
    )
    assert reloaded is not None
    assert reloaded.workspace_revision == 2
    stale = created.model_copy(update={"workspace_revision": 3})
    assert service.replace_workspace_if_match(created, stale) is False


def test_ready_guidance_not_appended_for_daily_turns() -> None:
    presenter = ConversationSetupOnboardingPresenter()
    snapshot = WorkspaceSetupSnapshotV1(
        workspace_id=_WORKSPACE,
        host_ready=True,
        phase=SetupPhaseV1.READY,
        can_ask=True,
        has_usable_knowledge=True,
        sync_in_progress=False,
        attention_required=False,
        knowledge_summary=SetupKnowledgeSummaryV1(
            total=1,
            usable=1,
            indexed=1,
            live=0,
            active=1,
            disabled=0,
            attention_required=0,
        ),
        next_action=SetupNextActionV1.ASK_QUESTION,
        suggested_question="What changed this week?",
        updated_at=_NOW,
    )
    assert presenter.should_append_snapshot_guidance(snapshot) is False
    guidance = presenter.render_snapshot_guidance(snapshot)
    assert "ready" in guidance.casefold()


@pytest.mark.asyncio
async def test_expired_confirmation_token_rejected() -> None:
    items = (_item(),)
    workspace = Workspace(
        workspace_id=_WORKSPACE,
        tenant_id=_TENANT,
        name="Alpha",
        status=WorkspaceStatus.ACTIVE,
        workspace_revision=1,
        created_at=_NOW,
        updated_at=_NOW,
    )
    expired = _CONFIRM_CODEC.issue(
        DestructiveActionConfirmationV1(
            token="",
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            action_kind=DestructiveActionKindV1.WORKSPACE_DELETE,
            target_id=_WORKSPACE,
            expected_state_version=1,
            expires_at=_NOW - timedelta(minutes=1),
        )
    )
    executor = ConversationInteractionExecutor(
        workspace_service=WorkspaceServiceFake(workspace),  # type: ignore[arg-type]
        workspace_selection_service=SimpleNamespace(),  # type: ignore[arg-type]
        destructive_confirmation_codec=_CONFIRM_CODEC,
        clock=lambda: _NOW,
    )
    result = await executor.execute(
        ConversationInteractionExecutionCommand(
            tenant_id=_TENANT,
            planning_request=ConversationPlanningRequest(message_text="confirm"),
            interaction_plan=ConversationInteractionPlan(
                plan_version="2",
                actions=(
                    DestructiveActionConfirmPlannedAction(
                        action_id="confirm",
                        action_type="destructive.confirm",
                        confirmation_token=expired,
                    ),
                ),
                response_mode="aggregate",
            ),
            execution_context=_context(),
        )
    )
    assert result.action_results[0].status is ConversationActionExecutionStatus.FAILED
