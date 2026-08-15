# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from intergrax.integrations.contracts.conversation_channel import (
    ConversationAttachmentReference,
)
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
from local_workspace_application.conversation.interaction_models import (
    ConversationInteractionPlan,
    KnowledgeAddAttachmentsPlannedAction,
    SourceCandidateAttachPlannedAction,
    WorkspaceActivatePlannedAction,
    WorkspaceAskPlannedAction,
    WorkspaceCreatePlannedAction,
    WorkspaceListPlannedAction,
    WorkspaceReference,
    WorkspaceReferenceKind,
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
    ResolvedConversationWorkspaceContextV1,
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
    def __init__(self, plan: ConversationInteractionPlan | None = None) -> None:
        self.plan_result = plan
        self.calls = 0

    async def plan(self, request, **_: object):
        self.calls += 1
        if self.plan_result is None:
            raise AssertionError("planner should not run")
        return self.plan_result


class _Executor:
    def __init__(
        self,
        result: ConversationInteractionExecutionResult | None = None,
    ) -> None:
        self.result = result
        self.calls = []

    async def execute(self, command):
        self.calls.append(command)
        if self.result is None:
            raise AssertionError("executor result not configured")
        return self.result


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
    planner: _Planner | None = None,
    executor: _Executor | None = None,
) -> ConversationInteractionApplicationService:
    store = InMemoryDocumentStore()
    return ConversationInteractionApplicationService(
        context_resolver=resolver,  # type: ignore[arg-type]
        planner=planner or _Planner(),  # type: ignore[arg-type]
        executor=executor or _Executor(),  # type: ignore[arg-type]
        renderer=ConversationInteractionResponseRenderer(),
        receipt_repository=ConversationInteractionEventReceiptRepository(store),
        workspace_service=_WorkspaceService(),
        personal_allowed_capabilities=frozenset(ConversationProductCapability),
        ingress_bootstrap_service=bootstrap,
        setup_snapshot_service=setup_snapshot,
        setup_onboarding_presenter=ConversationSetupOnboardingPresenter(),
        clock=lambda: _NOW,
    )


def _snapshot(
    *,
    phase: SetupPhaseV1,
    can_ask: bool,
    next_action: SetupNextActionV1,
    sync_in_progress: bool = False,
    attention_required: bool = False,
) -> WorkspaceSetupSnapshotV1:
    is_ready = phase is SetupPhaseV1.READY
    return WorkspaceSetupSnapshotV1(
        workspace_id="ws-1",
        host_ready=True,
        phase=phase,
        can_ask=can_ask,
        has_usable_knowledge=is_ready,
        sync_in_progress=sync_in_progress,
        attention_required=attention_required,
        knowledge_summary=SetupKnowledgeSummaryV1(
            total=1 if phase is not SetupPhaseV1.NO_KNOWLEDGE else 0,
            indexed=1 if phase is not SetupPhaseV1.NO_KNOWLEDGE else 0,
            live=1 if is_ready else 0,
            active=1 if is_ready else 0,
            disabled=0,
            attention_required=1 if attention_required else 0,
            usable=1 if is_ready else 0,
        ),
        next_action=next_action,
        updated_at=_NOW,
    )


def _resolved_context() -> ResolvedConversationWorkspaceContextV1:
    return ResolvedConversationWorkspaceContextV1(
        tenant_id="tenant-a",
        conversation_context_binding_id="binding-1",
        audience_mode=ConversationAudienceMode.PERSONAL,
        workspace_id="ws-1",
        principal_ref="U1",
        canonical_thread_ref="thread-1",
        activation_policy=ConversationActivationPolicy.ALWAYS,
        thread_context_policy=ConversationThreadContextPolicy.CURRENT_THREAD_BOUNDED,
    )


def _plan(*actions: object) -> ConversationInteractionPlan:
    return ConversationInteractionPlan(
        plan_version="2",
        actions=tuple(actions),
        response_mode="aggregate",
    )


def _active_workspace() -> WorkspaceReference:
    return WorkspaceReference(kind=WorkspaceReferenceKind.active)


def _execution_result(action_type: str) -> ConversationInteractionExecutionResult:
    now = _NOW
    return ConversationInteractionExecutionResult(
        execution_id="exec-1",
        tenant_id="tenant-a",
        plan_version="2",
        started_at=now,
        completed_at=now,
        status=ConversationInteractionOverallStatus.COMPLETED,
        action_results=(
            ConversationActionExecutionResult(
                action_id="action-1",
                action_type=action_type,
                status=ConversationActionExecutionStatus.COMPLETED,
                artifact=ConversationExecutionArtifact(
                    artifact_type=action_type,
                    data={"workspace_id": "ws-1"},
                ),
                started_at=now,
                completed_at=now,
            ),
        ),
    )


def _active_service(
    *,
    plan: ConversationInteractionPlan,
    snapshot: WorkspaceSetupSnapshotV1,
    action_type: str,
) -> tuple[
    ConversationInteractionApplicationService,
    MagicMock,
    _Planner,
    _Executor,
]:
    snapshot_service = MagicMock()
    snapshot_service.derive_snapshot.return_value = snapshot
    resolver = _Resolver()
    resolver.resolve = MagicMock(return_value=_resolved_context())
    planner = _Planner(plan)
    executor = _Executor(_execution_result(action_type))
    return (
        _service(
            resolver=resolver,
            setup_snapshot=snapshot_service,
            planner=planner,
            executor=executor,
        ),
        snapshot_service,
        planner,
        executor,
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
    snapshot_service = MagicMock()
    snapshot_service.derive_snapshot.return_value = _snapshot(
        phase=SetupPhaseV1.NO_KNOWLEDGE,
        can_ask=False,
        next_action=SetupNextActionV1.ADD_SOURCE,
    )
    resolver = _Resolver()
    resolver.resolve = MagicMock(return_value=_resolved_context())
    plan = _plan(
        WorkspaceAskPlannedAction(
            action_id="ask-1",
            action_type="workspace.ask",
            workspace=_active_workspace(),
            question="What is our leave policy?",
        )
    )
    planner = _Planner(plan)
    executor = _Executor()
    service = _service(
        resolver=resolver,
        setup_snapshot=snapshot_service,
        planner=planner,
        executor=executor,
    )

    command = ConversationInteractionApplicationCommand(
        tenant_id="tenant-a",
        ingress=_ingress(),
        message_text="What is our leave policy?",
    )
    result = await service.handle(command)
    assert result.should_send
    assert "cannot answer that yet" in result.response_text.casefold()
    assert planner.calls == 1
    assert executor.calls == []
    snapshot_service.derive_snapshot.assert_called_once_with(
        tenant_id="tenant-a",
        workspace_id="ws-1",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("phase", "next_action", "sync_in_progress", "attention_required"),
    (
        (
            SetupPhaseV1.SYNCING,
            SetupNextActionV1.WAIT_FOR_SYNC,
            True,
            False,
        ),
        (
            SetupPhaseV1.ATTENTION_REQUIRED,
            SetupNextActionV1.RETRY_OR_FIX_SOURCE,
            False,
            True,
        ),
    ),
)
async def test_question_is_gated_for_syncing_or_attention_snapshot(
    phase: SetupPhaseV1,
    next_action: SetupNextActionV1,
    sync_in_progress: bool,
    attention_required: bool,
) -> None:
    plan = _plan(
        WorkspaceAskPlannedAction(
            action_id="ask-1",
            action_type="workspace.ask",
            workspace=_active_workspace(),
            question="What is our leave policy?",
        )
    )
    service, _, planner, executor = _active_service(
        plan=plan,
        snapshot=_snapshot(
            phase=phase,
            can_ask=False,
            next_action=next_action,
            sync_in_progress=sync_in_progress,
            attention_required=attention_required,
        ),
        action_type="workspace.ask",
    )

    result = await service.handle(
        ConversationInteractionApplicationCommand(
            tenant_id="tenant-a",
            ingress=_ingress(),
            message_text="What is our leave policy?",
        )
    )

    assert result.should_send
    assert "cannot answer that yet" in result.response_text.casefold()
    assert planner.calls == 1
    assert executor.calls == []


@pytest.mark.asyncio
async def test_ready_question_is_planned_and_executed() -> None:
    plan = _plan(
        WorkspaceAskPlannedAction(
            action_id="ask-1",
            action_type="workspace.ask",
            workspace=_active_workspace(),
            question="What is our leave policy?",
        )
    )
    service, snapshot_service, planner, executor = _active_service(
        plan=plan,
        snapshot=_snapshot(
            phase=SetupPhaseV1.READY,
            can_ask=True,
            next_action=SetupNextActionV1.ASK_QUESTION,
        ),
        action_type="workspace.ask",
    )

    result = await service.handle(
        ConversationInteractionApplicationCommand(
            tenant_id="tenant-a",
            ingress=_ingress(),
            message_text="What is our leave policy?",
        )
    )

    assert result.should_send
    assert planner.calls == 1
    assert len(executor.calls) == 1
    assert snapshot_service.derive_snapshot.call_count == 2
    snapshot_service.derive_snapshot.assert_any_call(
        tenant_id="tenant-a",
        workspace_id="ws-1",
    )


@pytest.mark.asyncio
async def test_workspace_switch_is_executed_when_snapshot_is_not_ready() -> None:
    plan = _plan(
        WorkspaceActivatePlannedAction(
            action_id="switch-1",
            action_type="workspace.activate",
            workspace=WorkspaceReference(
                kind=WorkspaceReferenceKind.name,
                value="Beta",
            ),
        )
    )
    service, _, planner, executor = _active_service(
        plan=plan,
        snapshot=_snapshot(
            phase=SetupPhaseV1.NO_KNOWLEDGE,
            can_ask=False,
            next_action=SetupNextActionV1.ADD_SOURCE,
        ),
        action_type="workspace.activate",
    )

    await service.handle(
        ConversationInteractionApplicationCommand(
            tenant_id="tenant-a",
            ingress=_ingress(),
            message_text="switch to Beta",
        )
    )

    assert planner.calls == 1
    assert len(executor.calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "action",
    (
        WorkspaceListPlannedAction(
            action_id="list-1",
            action_type="workspace.list",
        ),
        WorkspaceCreatePlannedAction(
            action_id="create-1",
            action_type="workspace.create",
            name="Beta",
        ),
    ),
)
async def test_workspace_list_or_create_is_executed_when_snapshot_is_not_ready(
    action: object,
) -> None:
    service, _, planner, executor = _active_service(
        plan=_plan(action),
        snapshot=_snapshot(
            phase=SetupPhaseV1.NO_KNOWLEDGE,
            can_ask=False,
            next_action=SetupNextActionV1.ADD_SOURCE,
        ),
        action_type=action.action_type,  # type: ignore[attr-defined]
    )

    await service.handle(
        ConversationInteractionApplicationCommand(
            tenant_id="tenant-a",
            ingress=_ingress(),
            message_text="manage workspaces",
        )
    )

    assert planner.calls == 1
    assert len(executor.calls) == 1


@pytest.mark.asyncio
async def test_recovery_action_reaches_executor_when_attention_is_required() -> None:
    plan = _plan(
        SourceCandidateAttachPlannedAction(
            action_id="retry-1",
            action_type="source_candidate.attach",
            workspace=_active_workspace(),
            candidate_reference_kind="name",
            candidate_reference="Docs",
        )
    )
    service, _, planner, executor = _active_service(
        plan=plan,
        snapshot=_snapshot(
            phase=SetupPhaseV1.ATTENTION_REQUIRED,
            can_ask=False,
            next_action=SetupNextActionV1.RETRY_OR_FIX_SOURCE,
            attention_required=True,
        ),
        action_type="source_candidate.attach",
    )

    await service.handle(
        ConversationInteractionApplicationCommand(
            tenant_id="tenant-a",
            ingress=_ingress(),
            message_text="retry Docs",
        )
    )

    assert planner.calls == 1
    assert len(executor.calls) == 1


@pytest.mark.asyncio
async def test_attachment_intake_is_executed_when_snapshot_is_not_ready() -> None:
    attachment = ConversationAttachmentReference(
        attachment_id="att-1",
        file_name="notes.pdf",
        content_type="application/pdf",
        size_bytes=10,
    )
    plan = _plan(
        KnowledgeAddAttachmentsPlannedAction(
            action_id="attach-1",
            action_type="knowledge.add_attachments",
            workspace=_active_workspace(),
            attachment_ids=("att-1",),
        )
    )
    service, _, planner, executor = _active_service(
        plan=plan,
        snapshot=_snapshot(
            phase=SetupPhaseV1.NO_KNOWLEDGE,
            can_ask=False,
            next_action=SetupNextActionV1.ADD_SOURCE,
        ),
        action_type="knowledge.add_attachments",
    )

    await service.handle(
        ConversationInteractionApplicationCommand(
            tenant_id="tenant-a",
            ingress=_ingress(),
            message_text="add this file",
            attachments=(attachment,),
        )
    )

    assert planner.calls == 1
    assert len(executor.calls) == 1


@pytest.mark.asyncio
async def test_mixed_ask_plan_is_fail_closed_when_snapshot_is_not_ready() -> None:
    plan = _plan(
        WorkspaceAskPlannedAction(
            action_id="ask-1",
            action_type="workspace.ask",
            workspace=_active_workspace(),
            question="What is our leave policy?",
        ),
        WorkspaceActivatePlannedAction(
            action_id="switch-1",
            action_type="workspace.activate",
            workspace=WorkspaceReference(
                kind=WorkspaceReferenceKind.name,
                value="Beta",
            ),
        ),
    )
    service, _, planner, executor = _active_service(
        plan=plan,
        snapshot=_snapshot(
            phase=SetupPhaseV1.NO_KNOWLEDGE,
            can_ask=False,
            next_action=SetupNextActionV1.ADD_SOURCE,
        ),
        action_type="workspace.ask",
    )

    result = await service.handle(
        ConversationInteractionApplicationCommand(
            tenant_id="tenant-a",
            ingress=_ingress(),
            message_text="answer and switch",
        )
    )

    assert "cannot answer that yet" in result.response_text.casefold()
    assert planner.calls == 1
    assert executor.calls == []


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
