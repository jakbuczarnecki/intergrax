from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.integrations.contracts.conversation_channel import (
    ConversationActor,
    ConversationAddress,
    ConversationAttachmentReference,
    ConversationEventKind,
    InboundConversationEvent,
    OutboundConversationMessage,
)
from local_workspace_application.slack_companion.authorization import (
    SlackCompanionAuthConfig,
)
from local_workspace_application.slack_companion.dedupe_repository import (
    SlackEventDedupeRepository,
)
from local_workspace_application.slack_companion.workflow import SlackAskWorkflow
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
    ConversationPlanningRequest,
    WorkspaceListPlannedAction,
)
from local_workspace_application.workspaces.conversation_context_models import (
    ConversationActivationPolicy,
    ConversationAudienceMode,
    ConversationIngressContextV1,
    ConversationObservedAudience,
    ConversationProductCapability,
    ConversationThreadContextPolicy,
    ResolvedConversationWorkspaceContextV1,
)


class _Resolver:
    def resolve(self, *, tenant_id: str, ingress: ConversationIngressContextV1):
        assert tenant_id == "tenant-a"
        assert ingress.observed_audience is ConversationObservedAudience.PERSONAL
        return ResolvedConversationWorkspaceContextV1(
            tenant_id=tenant_id,
            conversation_context_binding_id="binding-1",
            audience_mode=ConversationAudienceMode.PERSONAL,
            workspace_id="workspace-1",
            principal_ref="U1",
            canonical_thread_ref=ingress.opaque_thread_ref,
            activation_policy=ConversationActivationPolicy.ALWAYS,
            thread_context_policy=ConversationThreadContextPolicy.CURRENT_THREAD_BOUNDED,
        )


class _WorkspaceService:
    def list_workspaces(self, *, tenant_id: str):
        assert tenant_id == "tenant-a"
        return [SimpleNamespace(workspace_id="workspace-1", name="Project Alfa")]


class _Planner:
    def __init__(self) -> None:
        self.calls = 0
        self.requests: list[ConversationPlanningRequest] = []

    async def plan(self, request: ConversationPlanningRequest, **_: object):
        self.calls += 1
        self.requests.append(request)
        return ConversationInteractionPlan(
            plan_version="2",
            actions=(
                WorkspaceListPlannedAction(
                    action_id="list-1",
                    action_type="workspace.list",
                ),
            ),
            response_mode="aggregate",
        )


class _Executor:
    def __init__(self) -> None:
        self.calls = 0
        self.commands = []

    async def execute(self, command):
        self.calls += 1
        self.commands.append(command)
        now = datetime.now(UTC)
        return ConversationInteractionExecutionResult(
            execution_id=command.execution_id or "execution-1",
            tenant_id=command.tenant_id,
            plan_version="2",
            started_at=now,
            completed_at=now,
            status=ConversationInteractionOverallStatus.COMPLETED,
            action_results=(
                ConversationActionExecutionResult(
                    action_id="list-1",
                    action_type="workspace.list",
                    status=ConversationActionExecutionStatus.COMPLETED,
                    artifact=ConversationExecutionArtifact(
                        artifact_type="workspace.list",
                        data={"workspaces": [{"name": "Project Alfa"}]},
                    ),
                    started_at=now,
                    completed_at=now,
                ),
            ),
        )


class _Renderer:
    def __init__(self) -> None:
        self.calls = 0

    def render(self, result: ConversationInteractionExecutionResult) -> str:
        self.calls += 1
        assert result.status is ConversationInteractionOverallStatus.COMPLETED
        return "Workspaces: 1"


class _InteractionAdapterFake:
    def __init__(self) -> None:
        self.calls = 0
        self.sent = 0

    async def handle(self, command):
        self.calls += 1
        assert command.ingress.observed_audience is ConversationObservedAudience.PERSONAL
        return SimpleNamespace(
            should_send=True,
            response_text="Workspaces: 1",
            receipt=None,
        )

    def mark_response_sent(self, result) -> None:
        del result
        self.sent += 1

    def mark_response_failed(self, result) -> None:
        del result


class _SlackSender:
    def __init__(self) -> None:
        self.messages: list[OutboundConversationMessage] = []

    async def __call__(self, message: OutboundConversationMessage) -> None:
        self.messages.append(message)


def _slack_event(*, audience_metadata: str) -> InboundConversationEvent:
    return InboundConversationEvent(
        event_id="event-adapter-1",
        address=ConversationAddress(
            installation_id="T1",
            conversation_id="D1",
            thread_id="thread-1",
        ),
        actor=ConversationActor(actor_id="U1", is_bot=False),
        kind=ConversationEventKind.MESSAGE,
        text="list workspaces",
        metadata={"slack_channel_type": audience_metadata},
    )


def _command(*, event_ref: str = "event-1") -> ConversationInteractionApplicationCommand:
    return ConversationInteractionApplicationCommand(
        tenant_id="tenant-a",
        ingress=ConversationIngressContextV1(
            conversation_connection_ref="slack",
            opaque_conversation_ref="D1",
            opaque_thread_ref="thread-1",
            actor_principal_ref="U1",
            observed_audience=ConversationObservedAudience.PERSONAL,
            activation_signal="ordinary_message",
            provider_event_ref=event_ref,
        ),
        message_text="list workspaces",
        attachments=(
            ConversationAttachmentReference(
                attachment_id="F1",
                file_name="safe.txt",
                content_type="text/plain",
                size_bytes=4,
            ),
        ),
    )


@pytest.mark.asyncio
async def test_application_service_plans_executes_renders_once_and_deduplicates() -> None:
    planner = _Planner()
    executor = _Executor()
    renderer = _Renderer()
    service = ConversationInteractionApplicationService(
        context_resolver=_Resolver(),  # type: ignore[arg-type]
        planner=planner,  # type: ignore[arg-type]
        executor=executor,  # type: ignore[arg-type]
        renderer=renderer,  # type: ignore[arg-type]
        receipt_repository=ConversationInteractionEventReceiptRepository(
            InMemoryDocumentStore()
        ),
        workspace_service=_WorkspaceService(),
        personal_allowed_capabilities=frozenset(ConversationProductCapability),
    )

    first = await service.handle(_command())
    service.mark_response_sent(first)
    duplicate = await service.handle(_command())

    assert first.should_send is True
    assert first.response_text == "Workspaces: 1"
    assert duplicate.should_send is False
    assert planner.calls == 1
    assert executor.calls == 1
    assert renderer.calls == 1
    assert planner.requests[0].active_workspace_id == "workspace-1"
    assert planner.requests[0].available_workspaces[0].is_active is True
    assert executor.commands[0].planning_request is planner.requests[0]


@pytest.mark.asyncio
async def test_shared_audience_is_rejected_before_planning() -> None:
    planner = _Planner()
    executor = _Executor()
    renderer = _Renderer()
    service = ConversationInteractionApplicationService(
        context_resolver=_Resolver(),  # type: ignore[arg-type]
        planner=planner,  # type: ignore[arg-type]
        executor=executor,  # type: ignore[arg-type]
        renderer=renderer,  # type: ignore[arg-type]
        receipt_repository=ConversationInteractionEventReceiptRepository(
            InMemoryDocumentStore()
        ),
        workspace_service=_WorkspaceService(),
        personal_allowed_capabilities=frozenset(ConversationProductCapability),
    )
    command = _command(event_ref="event-shared").model_copy(
        update={
            "ingress": _command(event_ref="event-shared").ingress.model_copy(
                update={"observed_audience": ConversationObservedAudience.SHARED}
            )
        }
    )

    result = await service.handle(command)

    assert result.should_send is True
    assert result.execution_result.error is not None
    assert result.execution_result.error.code == "conversation_audience_not_supported"
    assert planner.calls == 0
    assert executor.calls == 0


@pytest.mark.asyncio
async def test_slack_adapter_uses_one_interaction_response_and_rejects_shared() -> None:
    adapter = _InteractionAdapterFake()
    sender = _SlackSender()
    workflow = SlackAskWorkflow(
        auth_config=SlackCompanionAuthConfig(
            approved_team_id="T1",
            approved_user_id="U1",
            tenant_id="tenant-a",
            active_workspace_id="workspace-1",
        ),
        dedupe=SlackEventDedupeRepository(InMemoryDocumentStore()),
        ask_client=SimpleNamespace(),  # type: ignore[arg-type]
        send=sender,
        interaction_application_service=adapter,  # type: ignore[arg-type]
    )

    await workflow.handle(_slack_event(audience_metadata="im"))
    assert adapter.calls == 1
    assert adapter.sent == 1
    assert [message.text for message in sender.messages] == ["Workspaces: 1"]

    await workflow.handle(
        _slack_event(audience_metadata="group").model_copy(
            update={"event_id": "event-adapter-shared"}
        )
    )
    assert adapter.calls == 1
    assert len(sender.messages) == 1
