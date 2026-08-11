# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from local_workspace_application.conversation.conversation_ingress_bootstrap import (
    ConversationIngressBootstrapService,
)
from local_workspace_application.conversation.conversation_setup_onboarding import (
    ConversationSetupOnboardingPresenter,
)
from local_workspace_application.conversation.interaction_application_service import (
    ConversationInteractionApplicationCommand,
    ConversationInteractionApplicationService,
)
from local_workspace_application.conversation.interaction_event_receipt import (
    ConversationInteractionEventReceiptRepository,
)
from local_workspace_application.conversation.interaction_execution_models import (
    ConversationActionExecutionResult,
    ConversationActionExecutionStatus,
    ConversationExecutionArtifact,
    ConversationInteractionExecutionResult,
    ConversationInteractionOverallStatus,
)
from local_workspace_application.conversation.interaction_response_renderer import (
    ConversationInteractionResponseRenderer,
)
from local_workspace_application.workspaces.conversation_context_models import (
    ConversationActivationPolicy,
    ConversationActivationSignal,
    ConversationIngressContextV1,
    ConversationObservedAudience,
    ConversationAudienceMode,
    ConversationExecutionContextV1,
    ConversationProductCapability,
    ConversationThreadContextPolicy,
)
from local_workspace_application.workspaces.conversation_context_resolution import (
    ConversationContextResolutionError,
)
from local_workspace_application.workspaces.workspace_setup_snapshot_service import (
    SetupKnowledgeSummaryV1,
    SetupNextActionV1,
    SetupPhaseV1,
    WorkspaceSetupSnapshotV1,
)

pytestmark = pytest.mark.unit

_NOW = datetime(2026, 8, 11, 8, 0, tzinfo=UTC)


class _Resolver:
    def __init__(self, *, error_code: str | None = None) -> None:
        self.error_code = error_code
        self.calls = 0

    def resolve(self, *, tenant_id: str, ingress: ConversationIngressContextV1):
        self.calls += 1
        if self.error_code:
            raise ConversationContextResolutionError(self.error_code)
        raise AssertionError("resolve should not succeed in these tests")


class _WorkspaceService:
    def list_workspaces(self, *, tenant_id: str):
        return [SimpleNamespace(workspace_id="ws-1", name="Alpha")]


class _Planner:
    async def plan(self, request, **_: object):
        raise AssertionError("planner should not run")


class _Executor:
    async def execute(self, command):
        raise AssertionError("executor should not run")


def _ingress() -> ConversationIngressContextV1:
    return ConversationIngressContextV1(
        conversation_connection_ref="slack",
        opaque_conversation_ref="D123",
        opaque_thread_ref="thread-1",
        actor_principal_ref="U1",
        observed_audience=ConversationObservedAudience.PERSONAL,
        activation_signal=ConversationActivationSignal.ORDINARY_MESSAGE,
        provider_event_ref="evt-1",
    )


def _service(
    *,
    resolver: _Resolver,
    setup_snapshot: MagicMock | None = None,
    bootstrap: ConversationIngressBootstrapService | None = None,
) -> ConversationInteractionApplicationService:
    store = InMemoryDocumentStore()
    return ConversationInteractionApplicationService(
        context_resolver=resolver,  # type: ignore[arg-type]
        planner=_Planner(),  # type: ignore[arg-type]
        executor=_Executor(),  # type: ignore[arg-type]
        renderer=ConversationInteractionResponseRenderer(),
        receipt_repository=ConversationInteractionEventReceiptRepository(store),
        workspace_service=_WorkspaceService(),
        personal_allowed_capabilities=frozenset(ConversationProductCapability),
        ingress_bootstrap_service=bootstrap,
        setup_snapshot_service=setup_snapshot,
        setup_onboarding_presenter=ConversationSetupOnboardingPresenter(),
        clock=lambda: _NOW,
    )


def _snapshot_ready() -> WorkspaceSetupSnapshotV1:
    return WorkspaceSetupSnapshotV1(
        workspace_id="ws-1",
        host_ready=True,
        phase=SetupPhaseV1.NO_KNOWLEDGE,
        can_ask=False,
        has_usable_knowledge=False,
        sync_in_progress=False,
        attention_required=False,
        knowledge_summary=SetupKnowledgeSummaryV1(
            total=0,
            indexed=0,
            live=0,
            active=0,
            disabled=0,
            attention_required=0,
            usable=0,
        ),
        next_action=SetupNextActionV1.ADD_SOURCE,
        updated_at=_NOW,
    )


@pytest.mark.asyncio
async def test_first_dm_without_workspace_selection_shows_welcome() -> None:
    service = _service(resolver=_Resolver(error_code="PERSONAL_WORKSPACE_SELECTION_MISSING"))
    command = ConversationInteractionApplicationCommand(
        tenant_id="tenant-a",
        ingress=_ingress(),
        message_text="hello",
    )
    result = await service.handle(command)
    assert result.should_send
    assert "Welcome to LKW" in result.response_text
    assert "Alpha" in result.response_text


@pytest.mark.asyncio
async def test_question_gated_when_snapshot_not_ready_for_ask() -> None:
    from local_workspace_application.workspaces.conversation_context_models import (
        ConversationAudienceMode,
        ConversationProductCapability,
        ConversationThreadContextPolicy,
        ResolvedConversationWorkspaceContextV1,
    )

    snapshot_service = MagicMock()
    snapshot_service.derive_snapshot.return_value = _snapshot_ready()
    resolver = _Resolver()
    resolver.resolve = MagicMock(
        return_value=ResolvedConversationWorkspaceContextV1(
            tenant_id="tenant-a",
            conversation_context_binding_id="binding-1",
            audience_mode=ConversationAudienceMode.PERSONAL,
            workspace_id="ws-1",
            principal_ref="U1",
            canonical_thread_ref="thread-1",
            activation_policy=ConversationActivationPolicy.ALWAYS,
            thread_context_policy=ConversationThreadContextPolicy.CURRENT_THREAD_BOUNDED,
        )
    )

    service = ConversationInteractionApplicationService(
        context_resolver=resolver,  # type: ignore[arg-type]
        planner=_Planner(),  # type: ignore[arg-type]
        executor=_Executor(),  # type: ignore[arg-type]
        renderer=ConversationInteractionResponseRenderer(),
        receipt_repository=ConversationInteractionEventReceiptRepository(
            InMemoryDocumentStore()
        ),
        workspace_service=_WorkspaceService(),
        personal_allowed_capabilities=frozenset(ConversationProductCapability),
        ingress_bootstrap_service=None,
        setup_snapshot_service=snapshot_service,
        setup_onboarding_presenter=ConversationSetupOnboardingPresenter(),
        clock=lambda: _NOW,
    )

    command = ConversationInteractionApplicationCommand(
        tenant_id="tenant-a",
        ingress=_ingress(),
        message_text="What is our leave policy?",
    )
    result = await service.handle(command)
    assert result.should_send
    assert "cannot answer that yet" in result.response_text.casefold()
    snapshot_service.derive_snapshot.assert_called()


@pytest.mark.asyncio
async def test_finish_success_appends_snapshot_guidance_after_action() -> None:
    store = InMemoryDocumentStore()
    receipt_repo = ConversationInteractionEventReceiptRepository(store)
    snapshot_service = MagicMock()
    snapshot_service.derive_snapshot.return_value = WorkspaceSetupSnapshotV1(
        workspace_id="ws-1",
        host_ready=True,
        phase=SetupPhaseV1.SYNCING,
        can_ask=False,
        has_usable_knowledge=False,
        sync_in_progress=True,
        attention_required=False,
        knowledge_summary=SetupKnowledgeSummaryV1(
            total=1,
            indexed=1,
            live=0,
            active=0,
            disabled=0,
            attention_required=0,
            usable=0,
        ),
        next_action=SetupNextActionV1.WAIT_FOR_SYNC,
        updated_at=_NOW,
    )
    service = ConversationInteractionApplicationService(
        context_resolver=MagicMock(),  # type: ignore[arg-type]
        planner=MagicMock(),  # type: ignore[arg-type]
        executor=MagicMock(),  # type: ignore[arg-type]
        renderer=ConversationInteractionResponseRenderer(),
        receipt_repository=receipt_repo,
        workspace_service=_WorkspaceService(),
        personal_allowed_capabilities=frozenset(ConversationProductCapability),
        setup_snapshot_service=snapshot_service,
        setup_onboarding_presenter=ConversationSetupOnboardingPresenter(),
        clock=lambda: _NOW,
    )
    claim = receipt_repo.claim(
        tenant_id="tenant-a",
        conversation_connection_ref="slack",
        provider_event_ref="evt-2",
        execution_id="exec-2",
    )
    now = _NOW
    execution_result = ConversationInteractionExecutionResult(
        execution_id="exec-2",
        tenant_id="tenant-a",
        plan_version="2",
        started_at=now,
        completed_at=now,
        status=ConversationInteractionOverallStatus.COMPLETED,
        action_results=(
            ConversationActionExecutionResult(
                action_id="attach-1",
                action_type="knowledge.add_attachments",
                status=ConversationActionExecutionStatus.COMPLETED,
                artifact=ConversationExecutionArtifact(
                    artifact_type="knowledge.add_attachments",
                    data={"attachments": [{"file_name": "notes.pdf"}]},
                ),
                started_at=now,
                completed_at=now,
            ),
        ),
    )
    context = ConversationExecutionContextV1(
        tenant_id="tenant-a",
        conversation_context_binding_id="binding-1",
        audience_mode=ConversationAudienceMode.PERSONAL,
        workspace_id="ws-1",
        principal_ref="U1",
        canonical_thread_ref="thread-1",
        activation_policy=ConversationActivationPolicy.ALWAYS,
        thread_context_policy=ConversationThreadContextPolicy.CURRENT_THREAD_BOUNDED,
        allowed_product_capabilities=frozenset(ConversationProductCapability),
    )
    result = service._finish_success(
        command=ConversationInteractionApplicationCommand(
            tenant_id="tenant-a",
            ingress=_ingress(),
            message_text="upload file",
        ),
        execution_context=context,
        receipt=claim.receipt,
        execution_result=execution_result,
        setup_snapshot=snapshot_service.derive_snapshot.return_value,
    )
    assert "being prepared" in result.response_text.casefold()
    assert "notes.pdf" not in result.response_text
