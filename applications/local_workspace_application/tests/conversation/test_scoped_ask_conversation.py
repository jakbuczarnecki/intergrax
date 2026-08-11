# © Artur Czarnecki. All rights reserved.

"""Conversation path tests for source-scoped indexed Ask."""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

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
    KnowledgeAskTargetReference,
    KnowledgeTargetReferenceKind,
    WorkspaceAskPlannedAction,
    WorkspaceReference,
    WorkspaceReferenceKind,
)
from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from local_workspace_application.workspaces.ask_models import AskRunStatus, WorkspaceAskRun
from local_workspace_application.workspaces.ask_repository import WorkspaceAskRepository
from local_workspace_application.workspaces.ask_service import WorkspaceAskService
from local_workspace_application.workspaces.conversation_context_models import (
    ConversationActivationPolicy,
    ConversationAudienceMode,
    ConversationExecutionContextV1,
    ConversationProductCapability,
    ConversationThreadContextPolicy,
)
from local_workspace_application.workspaces.knowledge_ask_scope_resolver import (
    KnowledgeAskScopeResolver,
)
from local_workspace_application.workspaces.knowledge_indexed_source_lifecycle_service import (
    IndexedSourceLifecycleStateV1,
    IndexedSourceSyncStateV1,
)
from local_workspace_application.workspaces.knowledge_inspection_operations_service import (
    KnowledgeInspectionService,
    indexed_knowledge_item_id,
)
from local_workspace_application.workspaces.knowledge_live_access_service import (
    LiveAccessLifecycleStateV1,
)

pytestmark = pytest.mark.unit

_NOW = datetime(2026, 8, 11, 10, 0, tzinfo=UTC)


def _inspection_service() -> KnowledgeInspectionService:
    indexed_binding = "indexed-binding-a"
    live_binding = "live-binding-a"
    configuration = SimpleNamespace(
        tenant_id="tenant-a",
        workspace_id="workspace-a",
        indexed_sources=(
            SimpleNamespace(
                indexed_source_binding_id=indexed_binding,
                cached_safe_display_label="Project Drive",
            ),
        ),
        live_access_bindings=(
            SimpleNamespace(
                live_access_binding_id=live_binding,
                derived_safe_display_label="HR API",
                derived_provider_id="hr",
                derived_resource_type="folder",
                allowed_capability_ids=(),
            ),
        ),
        updated_at=_NOW,
    )

    class _Configuration:
        def get_configuration(self, *, tenant_id: str, workspace_id: str):
            _ = tenant_id
            if workspace_id != configuration.workspace_id:
                return None
            return configuration

    class _IndexedLifecycle:
        def get(self, **kwargs: object):
            _ = kwargs
            return SimpleNamespace(
                tenant_id="tenant-a",
                workspace_id="workspace-a",
                source_id="source-a",
                indexed_source_binding_id=indexed_binding,
                knowledge_source_binding_ref="binding-a",
                lifecycle_state=IndexedSourceLifecycleStateV1.ACTIVE,
                lifecycle_revision=1,
                enabled=True,
                detached=False,
                sync_state=IndexedSourceSyncStateV1.SUCCEEDED,
                last_successful_sync_at=_NOW,
                last_error_code=None,
                updated_at=_NOW,
            )

    class _LiveLifecycle:
        def get(self, command):
            _ = command
            return SimpleNamespace(
                tenant_id="tenant-a",
                workspace_id="workspace-a",
                live_access_binding_id=live_binding,
                connection_ref="connection-a",
                knowledge_source_binding_ref="live-a",
                lifecycle_state=LiveAccessLifecycleStateV1.ACTIVE,
                configuration_revision=1,
                enabled=True,
                detached=False,
                runtime_available=True,
                last_error_code=None,
                updated_at=_NOW,
            )

    return KnowledgeInspectionService(
        configuration_service=_Configuration(),  # type: ignore[arg-type]
        indexed_source_lifecycle_service=_IndexedLifecycle(),  # type: ignore[arg-type]
        live_access_lifecycle_service=_LiveLifecycle(),  # type: ignore[arg-type]
    )


def _workspace_service() -> SimpleNamespace:
    return SimpleNamespace(
        get_workspace=lambda **kwargs: SimpleNamespace(workspace_id="workspace-a"),
        list_workspaces=lambda tenant_id: [SimpleNamespace(workspace_id="workspace-a")],
        repository=SimpleNamespace(document_store=SimpleNamespace()),
    )


def _command(action: WorkspaceAskPlannedAction) -> ConversationInteractionExecutionCommand:
    return ConversationInteractionExecutionCommand(
        tenant_id="tenant-a",
        planning_request=ConversationPlanningRequest(
            message_text="ask source 1",
            active_workspace_id="workspace-a",
            available_workspaces=(
                ConversationPlanningWorkspace(
                    workspace_id="workspace-a",
                    name="Workspace A",
                    is_active=True,
                ),
            ),
        ),
        execution_context=ConversationExecutionContextV1(
            tenant_id="tenant-a",
            conversation_context_binding_id="binding-1",
            audience_mode=ConversationAudienceMode.PERSONAL,
            workspace_id="workspace-a",
            principal_ref="principal-1",
            canonical_thread_ref="thread-1",
            activation_policy=ConversationActivationPolicy.ALWAYS,
            thread_context_policy=ConversationThreadContextPolicy.CURRENT_THREAD_BOUNDED,
            allowed_product_capabilities=frozenset(
                {ConversationProductCapability.READ_ONLY_ASK}
            ),
        ),
        interaction_plan=ConversationInteractionPlan(
            plan_version="2",
            actions=(action,),
            clarifications=(),
            response_mode="aggregate",
        ),
    )


@pytest.mark.asyncio
async def test_executor_resolves_ordinal_target_for_scoped_ask() -> None:
    ask_service = SimpleNamespace()
    ask_service.ask = AsyncMock(
        return_value=WorkspaceAskRun(
            run_id="run-1",
            tenant_id="tenant-a",
            workspace_id="workspace-a",
            question="q",
            knowledge_item_ids=(indexed_knowledge_item_id("indexed-binding-a"),),
            status=AskRunStatus.COMPLETED,
            answer="ok",
            citations=[],
            created_at=_NOW,
            completed_at=_NOW,
        )
    )
    executor = ConversationInteractionExecutor(
        workspace_service=_workspace_service(),  # type: ignore[arg-type]
        workspace_selection_service=SimpleNamespace(),  # type: ignore[arg-type]
        ask_service=ask_service,  # type: ignore[arg-type]
        knowledge_inspection_service=_inspection_service(),
    )
    action = WorkspaceAskPlannedAction(
        action_id="action-1",
        action_type="workspace.ask",
        workspace=WorkspaceReference(kind=WorkspaceReferenceKind.active, value=None),
        question="What is the policy?",
        knowledge_targets=(
            KnowledgeAskTargetReference(
                target_reference_kind=KnowledgeTargetReferenceKind.ordinal,
                target_reference="1",
            ),
        ),
    )
    result = await executor.execute(_command(action))
    action_result = result.action_results[0]
    assert action_result.status == ConversationActionExecutionStatus.COMPLETED, (
        action_result.error.code if action_result.error else None
    )
    ask_service.ask.assert_awaited_once()
    kwargs = ask_service.ask.await_args.kwargs
    assert kwargs["knowledge_scope"].knowledge_item_ids == (
        indexed_knowledge_item_id("indexed-binding-a"),
    )


@pytest.mark.asyncio
async def test_live_source_selected_returns_safe_failed_ask() -> None:
    inspection = _inspection_service()
    store = InMemoryDocumentStore()
    workspace_service = SimpleNamespace(
        get_workspace=lambda **kwargs: SimpleNamespace(workspace_id="workspace-a"),
        list_workspaces=lambda tenant_id: [SimpleNamespace(workspace_id="workspace-a")],
        repository=SimpleNamespace(document_store=store),
    )
    repo = WorkspaceAskRepository(store)
    ask_service = WorkspaceAskService(
        workspace_service=workspace_service,  # type: ignore[arg-type]
        workspace_repository=workspace_service.repository,  # type: ignore[arg-type]
        ask_repository=repo,
        task_executor=SimpleNamespace(execute=AsyncMock()),  # type: ignore[arg-type]
        scope_resolver=KnowledgeAskScopeResolver(inspection),
    )
    executor = ConversationInteractionExecutor(
        workspace_service=workspace_service,  # type: ignore[arg-type]
        workspace_selection_service=SimpleNamespace(),  # type: ignore[arg-type]
        ask_service=ask_service,
        knowledge_inspection_service=inspection,
    )
    action = WorkspaceAskPlannedAction(
        action_id="action-1",
        action_type="workspace.ask",
        workspace=WorkspaceReference(kind=WorkspaceReferenceKind.active, value=None),
        question="What is HR policy?",
        knowledge_targets=(
            KnowledgeAskTargetReference(
                target_reference_kind=KnowledgeTargetReferenceKind.ordinal,
                target_reference="2",
            ),
        ),
    )
    result = await executor.execute(_command(action))
    action_result = result.action_results[0]
    assert action_result.status == ConversationActionExecutionStatus.COMPLETED, (
        action_result.error.code if action_result.error else None
    )
    artifact = action_result.artifact
    assert artifact is not None
    assert artifact.data["status"] == AskRunStatus.FAILED
