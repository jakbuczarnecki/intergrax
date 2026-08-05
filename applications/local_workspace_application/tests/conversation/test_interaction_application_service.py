from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

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
    ConversationEventReceipt,
    ConversationEventReceiptError,
    ConversationEventReceiptStatus,
    ConversationInteractionEventReceiptRepository,
    _MAX_RESPONSE_LENGTH,
)
from local_workspace_application.conversation.interaction_execution_models import (
    ConversationActionExecutionResult,
    ConversationActionExecutionStatus,
    ConversationExecutionArtifact,
    ConversationInteractionExecutionResult,
    ConversationInteractionOverallStatus,
)
from local_workspace_application.conversation.interaction_response_renderer import (
    MAX_RESPONSE_CHARS,
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


def _receipt(
    *,
    status: ConversationEventReceiptStatus = ConversationEventReceiptStatus.PROCESSING,
    safe_response: str | None = None,
    completed_at: datetime | None = None,
    created_at: datetime | None = None,
) -> ConversationEventReceipt:
    response_hash = (
        hashlib.sha256(safe_response.encode("utf-8")).hexdigest()
        if safe_response is not None
        else None
    )
    return ConversationEventReceipt(
        tenant_id="tenant-a",
        conversation_connection_ref="slack",
        provider_event_ref="event-1",
        status=status,
        execution_id="execution-1",
        safe_response=safe_response,
        response_hash=response_hash,
        created_at=created_at or datetime.now(UTC),
        completed_at=completed_at,
    )


def _claimed_receipt(
    repository: ConversationInteractionEventReceiptRepository,
) -> ConversationEventReceipt:
    return repository.claim(
        tenant_id="tenant-a",
        conversation_connection_ref="slack",
        provider_event_ref="event-1",
        execution_id="execution-1",
    ).receipt


def test_receipt_accepts_valid_processing_and_normalizes_identity() -> None:
    receipt = ConversationEventReceipt(
        tenant_id=" tenant-a ",
        conversation_connection_ref=" slack ",
        provider_event_ref=" event-1 ",
        status=ConversationEventReceiptStatus.PROCESSING,
        execution_id=" execution-1 ",
        created_at=datetime.now(UTC),
    )

    assert receipt.tenant_id == "tenant-a"
    assert receipt.conversation_connection_ref == "slack"
    assert receipt.provider_event_ref == "event-1"
    assert receipt.execution_id == "execution-1"


def test_receipt_rejects_processing_response_and_blank_identity() -> None:
    with pytest.raises(ValidationError):
        ConversationEventReceipt(
            tenant_id="tenant-a",
            conversation_connection_ref="slack",
            provider_event_ref="event-1",
            status=ConversationEventReceiptStatus.PROCESSING,
            execution_id="execution-1",
            safe_response="response",
            response_hash=hashlib.sha256(b"response").hexdigest(),
            created_at=datetime.now(UTC),
        )

    with pytest.raises(ValidationError):
        ConversationEventReceipt(
            tenant_id="   ",
            conversation_connection_ref="slack",
            provider_event_ref="event-1",
            status=ConversationEventReceiptStatus.PROCESSING,
            execution_id="execution-1",
            created_at=datetime.now(UTC),
        )


def test_receipt_rejects_incomplete_or_inconsistent_response_states() -> None:
    now = datetime.now(UTC)
    with pytest.raises(ValidationError):
        _receipt(status=ConversationEventReceiptStatus.RESPONSE_PENDING)
    with pytest.raises(ValidationError):
        ConversationEventReceipt(
            tenant_id="tenant-a",
            conversation_connection_ref="slack",
            provider_event_ref="event-1",
            status=ConversationEventReceiptStatus.RESPONSE_PENDING,
            execution_id="execution-1",
            safe_response="response",
            response_hash="wrong",
            created_at=now,
            completed_at=now,
        )
    with pytest.raises(ValidationError):
        _receipt(
            status=ConversationEventReceiptStatus.RESPONSE_SENT,
            safe_response="response",
        )


@pytest.mark.parametrize(
    "timestamp",
    (
        datetime.now(),
        datetime.now(timezone(timedelta(hours=2))),
    ),
)
def test_receipt_rejects_naive_and_non_utc_timestamps(
    timestamp: datetime,
) -> None:
    with pytest.raises(ValidationError):
        _receipt(created_at=timestamp)


def test_receipt_model_rejects_oversized_response() -> None:
    response = "x" * (_MAX_RESPONSE_LENGTH + 1)

    with pytest.raises(ValidationError):
        ConversationEventReceipt(
            tenant_id="tenant-a",
            conversation_connection_ref="slack",
            provider_event_ref="event-1",
            status=ConversationEventReceiptStatus.RESPONSE_PENDING,
            execution_id="execution-1",
            safe_response=response,
            response_hash=hashlib.sha256(response.encode("utf-8")).hexdigest(),
            created_at=datetime.now(UTC),
            completed_at=datetime.now(UTC),
        )


def test_receipt_pending_transition_revalidates_and_truncates_defensively() -> None:
    repository = ConversationInteractionEventReceiptRepository(
        InMemoryDocumentStore()
    )
    processing = _claimed_receipt(repository)
    pending = repository.mark_response_pending(
        receipt=processing,
        response="  " + "safe " * _MAX_RESPONSE_LENGTH,
    )

    assert pending.status is ConversationEventReceiptStatus.RESPONSE_PENDING
    assert pending.safe_response is not None
    assert len(pending.safe_response) == _MAX_RESPONSE_LENGTH
    assert pending.response_hash == hashlib.sha256(
        pending.safe_response.encode("utf-8")
    ).hexdigest()
    assert pending.completed_at is not None
    assert repository.claim(
        tenant_id="tenant-a",
        conversation_connection_ref="slack",
        provider_event_ref="event-1",
        execution_id="other-execution",
    ).receipt == pending


def test_receipt_rejects_empty_pending_response() -> None:
    repository = ConversationInteractionEventReceiptRepository(
        InMemoryDocumentStore()
    )

    with pytest.raises(ConversationEventReceiptError) as error:
        repository.mark_response_pending(
            receipt=_claimed_receipt(repository),
            response=" \n\t",
        )

    assert error.value.error_code == "conversation_receipt_response_empty"


def test_receipt_sent_and_failed_transitions_preserve_response_fields() -> None:
    repository = ConversationInteractionEventReceiptRepository(
        InMemoryDocumentStore()
    )
    pending = repository.mark_response_pending(
        receipt=_claimed_receipt(repository),
        response="safe response",
    )
    sent = repository.mark_response_sent(receipt=pending)

    assert sent.status is ConversationEventReceiptStatus.RESPONSE_SENT
    assert (
        sent.safe_response,
        sent.response_hash,
        sent.completed_at,
    ) == (
        pending.safe_response,
        pending.response_hash,
        pending.completed_at,
    )
    duplicate = repository.claim(
        tenant_id="tenant-a",
        conversation_connection_ref="slack",
        provider_event_ref="event-1",
        execution_id="other-execution",
    ).receipt
    assert duplicate.status is ConversationEventReceiptStatus.RESPONSE_SENT
    assert duplicate.safe_response == sent.safe_response
    assert duplicate.response_hash == sent.response_hash
    assert duplicate.completed_at == sent.completed_at

    repository = ConversationInteractionEventReceiptRepository(
        InMemoryDocumentStore()
    )
    pending = repository.mark_response_pending(
        receipt=_claimed_receipt(repository),
        response="safe response",
    )
    failed = repository.mark_response_failed(receipt=pending)

    assert failed.status is ConversationEventReceiptStatus.RESPONSE_FAILED
    assert (
        failed.safe_response,
        failed.response_hash,
        failed.completed_at,
    ) == (
        pending.safe_response,
        pending.response_hash,
        pending.completed_at,
    )
    duplicate = repository.claim(
        tenant_id="tenant-a",
        conversation_connection_ref="slack",
        provider_event_ref="event-1",
        execution_id="other-execution",
    ).receipt
    assert duplicate.status is ConversationEventReceiptStatus.RESPONSE_FAILED
    assert duplicate.safe_response == failed.safe_response
    assert duplicate.response_hash == failed.response_hash
    assert duplicate.completed_at == failed.completed_at


def test_receipt_invalid_transition_does_not_write() -> None:
    repository = ConversationInteractionEventReceiptRepository(
        InMemoryDocumentStore()
    )
    processing = _claimed_receipt(repository)

    with pytest.raises(ConversationEventReceiptError) as error:
        repository.mark_response_sent(receipt=processing)

    assert error.value.error_code == "conversation_receipt_transition_invalid"
    assert _claimed_receipt(repository).status is ConversationEventReceiptStatus.PROCESSING


def test_receipt_cas_conflict_preserves_winner() -> None:
    store = InMemoryDocumentStore()
    first_repository = ConversationInteractionEventReceiptRepository(store)
    second_repository = ConversationInteractionEventReceiptRepository(store)
    processing = _claimed_receipt(first_repository)
    pending = first_repository.mark_response_pending(
        receipt=processing,
        response="safe response",
    )
    winner = second_repository.mark_response_sent(receipt=pending)

    with pytest.raises(ConversationEventReceiptError) as error:
        first_repository.mark_response_failed(receipt=pending)

    assert error.value.error_code == "conversation_receipt_update_conflict"
    assert _claimed_receipt(first_repository) == winner


def test_malformed_stored_receipt_is_rejected_safely(monkeypatch) -> None:
    store = InMemoryDocumentStore()
    repository = ConversationInteractionEventReceiptRepository(store)
    monkeypatch.setattr(
        store,
        "get",
        lambda *_: SimpleNamespace(
            data={
                "tenant_id": "tenant-a",
                "conversation_connection_ref": "slack",
                "provider_event_ref": "event-1",
                "status": "response_sent",
                "execution_id": "execution-1",
                "safe_response": "response",
                "response_hash": "wrong",
                "created_at": datetime.now(UTC).isoformat(),
                "completed_at": datetime.now(UTC).isoformat(),
            }
        ),
    )

    with pytest.raises(ConversationEventReceiptError) as error:
        repository._get(
            tenant_id="tenant-a",
            conversation_connection_ref="slack",
            provider_event_ref="event-1",
        )

    assert error.value.error_code == "conversation_receipt_malformed"


def test_renderer_bound_fits_receipt_bound() -> None:
    assert MAX_RESPONSE_CHARS <= _MAX_RESPONSE_LENGTH
