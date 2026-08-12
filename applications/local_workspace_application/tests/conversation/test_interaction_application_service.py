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
    ConversationEventMemoryStatus,
    ConversationEventReceiptStatus,
    ConversationInteractionEventReceiptRepository,
    _MAX_RESPONSE_LENGTH,
)
from local_workspace_application.conversation.conversation_thread_memory_service import (
    ConversationThreadMemoryService,
    ConversationThreadMemoryServiceError,
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
from local_workspace_application.workspaces.conversation_context_memory import (
    DocumentStoreThreadMemoryLifecyclePort,
    SessionHistorySnapshotConversationThreadMemoryAdapter,
)
from local_workspace_application.workspaces.conversation_context_models import (
    ConversationThreadMemoryLimitsV1,
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


class _CountingThreadMemory:
    def __init__(
        self,
        store: InMemoryDocumentStore,
        *,
        failures_remaining: int = 0,
        always_fail: bool = False,
    ) -> None:
        self.append_calls = 0
        self._failures_remaining = failures_remaining
        self._always_fail = always_fail
        self._now = datetime(2026, 8, 5, 8, 0, tzinfo=UTC)
        self._last_context = None
        self._service = ConversationThreadMemoryService(
            adapter=SessionHistorySnapshotConversationThreadMemoryAdapter(
                port=DocumentStoreThreadMemoryLifecyclePort(store),
            ),
            limits=ConversationThreadMemoryLimitsV1(
                max_messages=20,
                max_bytes=16 * 1024,
                max_age_seconds=24 * 60 * 60,
            ),
            clock=lambda: self._now,
        )

    def load_recent_turns(self, *, context, now):
        return self._service.load_recent_turns(context=context, now=now)

    def append_exchange(
        self,
        *,
        context,
        user_text: str,
        assistant_text: str,
        user_created_at: datetime,
        assistant_created_at: datetime,
        exchange_id: str,
    ):
        self.append_calls += 1
        self._last_context = context
        if self._always_fail or self._failures_remaining:
            if self._failures_remaining:
                self._failures_remaining -= 1
            raise ConversationThreadMemoryServiceError("memory_append_failed")
        return self._service.append_exchange(
            context=context,
            user_text=user_text,
            assistant_text=assistant_text,
            user_created_at=user_created_at,
            assistant_created_at=assistant_created_at,
            exchange_id=exchange_id,
        )

    def persisted_turns(self):
        assert self._last_context is not None
        return self._service.load_recent_turns(
            context=self._last_context,
            now=self._now,
        )


class _MarkerFailureReceiptRepository(ConversationInteractionEventReceiptRepository):
    def __init__(
        self,
        store: InMemoryDocumentStore,
        *,
        completion_failures: int = 0,
        failure_marker_failures: int = 0,
    ) -> None:
        super().__init__(store)
        self._completion_failures = completion_failures
        self._failure_marker_failures = failure_marker_failures

    def mark_memory_completed(self, *, receipt, revision_id):
        if self._completion_failures:
            self._completion_failures -= 1
            raise ConversationEventReceiptError(
                "conversation_receipt_update_conflict"
            )
        return super().mark_memory_completed(
            receipt=receipt,
            revision_id=revision_id,
        )

    def mark_memory_failed(self, *, receipt, error_code):
        if self._failure_marker_failures:
            self._failure_marker_failures -= 1
            raise ConversationEventReceiptError(
                "conversation_receipt_update_conflict"
            )
        return super().mark_memory_failed(
            receipt=receipt,
            error_code=error_code,
        )


def _service_with_memory(
    *,
    repository: ConversationInteractionEventReceiptRepository,
    planner: _Planner,
    executor: _Executor,
    memory: _CountingThreadMemory,
) -> ConversationInteractionApplicationService:
    return ConversationInteractionApplicationService(
        context_resolver=_Resolver(),  # type: ignore[arg-type]
        planner=planner,  # type: ignore[arg-type]
        executor=executor,  # type: ignore[arg-type]
        renderer=_Renderer(),  # type: ignore[arg-type]
        receipt_repository=repository,
        workspace_service=_WorkspaceService(),
        personal_allowed_capabilities=frozenset(ConversationProductCapability),
        thread_memory_service=memory,  # type: ignore[arg-type]
        clock=lambda: datetime(2026, 8, 5, 8, 0, tzinfo=UTC),
    )


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
async def test_restart_reconstructs_recent_turns_before_planning() -> None:
    store = InMemoryDocumentStore()
    fixed_now = datetime(2026, 8, 5, 8, 0, tzinfo=UTC)

    def now() -> datetime:
        return fixed_now

    def build_service(planner: _Planner, executor: _Executor):
        memory = ConversationThreadMemoryService(
            adapter=SessionHistorySnapshotConversationThreadMemoryAdapter(
                port=DocumentStoreThreadMemoryLifecyclePort(store),
            ),
            limits=ConversationThreadMemoryLimitsV1(
                max_messages=20,
                max_bytes=16 * 1024,
                max_age_seconds=24 * 60 * 60,
            ),
            clock=now,
        )
        return ConversationInteractionApplicationService(
            context_resolver=_Resolver(),  # type: ignore[arg-type]
            planner=planner,  # type: ignore[arg-type]
            executor=executor,  # type: ignore[arg-type]
            renderer=_Renderer(),  # type: ignore[arg-type]
            receipt_repository=ConversationInteractionEventReceiptRepository(store),
            workspace_service=_WorkspaceService(),
            personal_allowed_capabilities=frozenset(ConversationProductCapability),
            thread_memory_service=memory,
            clock=now,
        )

    first_planner = _Planner()
    first_executor = _Executor()
    first_service = build_service(first_planner, first_executor)
    first = await first_service.handle(_command(event_ref="restart-1"))
    assert first.response_text == "Workspaces: 1"
    first_service.mark_response_sent(first)

    second_planner = _Planner()
    second_executor = _Executor()
    second_service = build_service(second_planner, second_executor)
    second_command = _command(event_ref="restart-2").model_copy(
        update={"message_text": "activate it"}
    )
    await second_service.handle(second_command)

    assert second_planner.calls == 1
    assert [
        (turn.role, turn.text) for turn in second_planner.requests[0].recent_turns
    ] == [
        ("user", "list workspaces"),
        ("assistant", "Workspaces: 1"),
    ]
    assert second_executor.calls == 1
    duplicate = await second_service.handle(_command(event_ref="restart-1"))
    assert duplicate.should_send is False
    assert second_planner.calls == 1


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
    memory_status: ConversationEventMemoryStatus = (
        ConversationEventMemoryStatus.NOT_REQUIRED
    ),
    memory_revision_id: str | None = None,
    memory_error_code: str | None = None,
    safe_user_memory_text: str | None = None,
) -> ConversationEventReceipt:
    response_hash = (
        hashlib.sha256(safe_response.encode("utf-8")).hexdigest()
        if safe_response is not None
        else None
    )
    if (
        memory_status is ConversationEventMemoryStatus.PENDING
        and safe_user_memory_text is None
    ):
        safe_user_memory_text = "list workspaces"
    return ConversationEventReceipt(
        tenant_id="tenant-a",
        conversation_connection_ref="slack",
        provider_event_ref="event-1",
        status=status,
        execution_id="execution-1",
        safe_response=safe_response,
        response_hash=response_hash,
        memory_status=memory_status,
        memory_revision_id=memory_revision_id,
        memory_error_code=memory_error_code,
        safe_user_memory_text=safe_user_memory_text,
        created_at=created_at or datetime.now(UTC),
        completed_at=completed_at,
    )


def _claimed_receipt(
    repository: ConversationInteractionEventReceiptRepository,
    *,
    event_ref: str = "event-1",
) -> ConversationEventReceipt:
    return repository.claim(
        tenant_id="tenant-a",
        conversation_connection_ref="slack",
        provider_event_ref=event_ref,
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


def test_receipt_rejects_sent_pending_memory() -> None:
    with pytest.raises(ValidationError):
        _receipt(
            status=ConversationEventReceiptStatus.RESPONSE_SENT,
            safe_response="response",
            memory_status=ConversationEventMemoryStatus.PENDING,
        )


@pytest.mark.parametrize(
    ("memory_status", "memory_revision_id", "memory_error_code"),
    (
        (ConversationEventMemoryStatus.NOT_REQUIRED, None, None),
        (ConversationEventMemoryStatus.COMPLETED, "revision-1", None),
        (ConversationEventMemoryStatus.FAILED, None, "memory_append_failed"),
    ),
)
def test_receipt_accepts_sent_terminal_memory(
    memory_status: ConversationEventMemoryStatus,
    memory_revision_id: str | None,
    memory_error_code: str | None,
) -> None:
    receipt = _receipt(
        status=ConversationEventReceiptStatus.RESPONSE_SENT,
        safe_response="response",
        memory_status=memory_status,
        memory_revision_id=memory_revision_id,
        memory_error_code=memory_error_code,
        completed_at=datetime.now(UTC),
    )

    assert receipt.status is ConversationEventReceiptStatus.RESPONSE_SENT


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


def test_receipt_sent_transition_rejects_pending_memory_without_write() -> None:
    repository = ConversationInteractionEventReceiptRepository(
        InMemoryDocumentStore()
    )
    pending = repository.mark_response_pending(
        receipt=_claimed_receipt(repository),
        response="safe response",
        memory_required=True,
        safe_user_memory_text="list workspaces",
    )

    with pytest.raises(ConversationEventReceiptError) as error:
        repository.mark_response_sent(receipt=pending)

    assert error.value.error_code == "conversation_receipt_memory_not_terminal"
    assert _claimed_receipt(repository) == pending


@pytest.mark.parametrize(
    "memory_status",
    (
        ConversationEventMemoryStatus.COMPLETED,
        ConversationEventMemoryStatus.FAILED,
    ),
)
def test_receipt_sent_transition_accepts_terminal_memory(
    memory_status: ConversationEventMemoryStatus,
) -> None:
    repository = ConversationInteractionEventReceiptRepository(
        InMemoryDocumentStore()
    )
    pending = repository.mark_response_pending(
        receipt=_claimed_receipt(repository),
        response="safe response",
        memory_required=True,
        safe_user_memory_text="list workspaces",
    )
    if memory_status is ConversationEventMemoryStatus.COMPLETED:
        pending = repository.mark_memory_completed(
            receipt=pending,
            revision_id="revision-1",
        )
    else:
        pending = repository.mark_memory_failed(
            receipt=pending,
            error_code="memory_append_failed",
        )

    sent = repository.mark_response_sent(receipt=pending)

    assert sent.status is ConversationEventReceiptStatus.RESPONSE_SENT
    assert sent.memory_status is memory_status


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


@pytest.mark.asyncio
async def test_memory_completion_marker_failure_recovers_without_reexecution() -> None:
    store = InMemoryDocumentStore()
    repository = _MarkerFailureReceiptRepository(store, completion_failures=1)
    memory = _CountingThreadMemory(store)
    planner = _Planner()
    executor = _Executor()
    service = _service_with_memory(
        repository=repository,
        planner=planner,
        executor=executor,
        memory=memory,
    )

    first = await service.handle(_command(event_ref="memory-marker-conflict"))
    assert first.receipt is not None
    assert first.receipt.memory_status is ConversationEventMemoryStatus.PENDING
    service.mark_response_sent(first)
    assert (
        _claimed_receipt(
            repository,
            event_ref="memory-marker-conflict",
        ).status
        is ConversationEventReceiptStatus.RESPONSE_PENDING
    )

    duplicate = await service.handle(_command(event_ref="memory-marker-conflict"))
    assert duplicate.receipt is not None
    assert (
        duplicate.receipt.memory_status
        is ConversationEventMemoryStatus.COMPLETED
    )
    service.mark_response_sent(duplicate)
    final = _claimed_receipt(
        repository,
        event_ref="memory-marker-conflict",
    )

    assert final.status is ConversationEventReceiptStatus.RESPONSE_SENT
    assert final.memory_status is ConversationEventMemoryStatus.COMPLETED
    assert len(memory.persisted_turns()) == 2
    assert memory.append_calls == 2
    assert planner.calls == 1
    assert executor.calls == 1


@pytest.mark.asyncio
async def test_memory_append_and_failure_marker_failure_remain_recoverable() -> None:
    store = InMemoryDocumentStore()
    repository = _MarkerFailureReceiptRepository(store, failure_marker_failures=1)
    memory = _CountingThreadMemory(store, failures_remaining=1)
    planner = _Planner()
    executor = _Executor()
    service = _service_with_memory(
        repository=repository,
        planner=planner,
        executor=executor,
        memory=memory,
    )

    first = await service.handle(_command(event_ref="memory-append-failure"))
    assert first.receipt is not None
    assert first.receipt.memory_status is ConversationEventMemoryStatus.PENDING
    assert first.response_text == "Workspaces: 1"
    service.mark_response_sent(first)
    assert (
        _claimed_receipt(
            repository,
            event_ref="memory-append-failure",
        ).memory_status
        is ConversationEventMemoryStatus.PENDING
    )

    duplicate = await service.handle(_command(event_ref="memory-append-failure"))
    assert duplicate.receipt is not None
    assert (
        duplicate.receipt.memory_status
        is ConversationEventMemoryStatus.COMPLETED
    )
    assert memory.append_calls == 2
    assert planner.calls == 1
    assert executor.calls == 1


@pytest.mark.asyncio
async def test_terminal_memory_failure_allows_sent_and_needs_no_duplicate_recovery() -> None:
    store = InMemoryDocumentStore()
    repository = ConversationInteractionEventReceiptRepository(store)
    memory = _CountingThreadMemory(store, always_fail=True)
    planner = _Planner()
    executor = _Executor()
    service = _service_with_memory(
        repository=repository,
        planner=planner,
        executor=executor,
        memory=memory,
    )

    first = await service.handle(_command(event_ref="memory-terminal-failure"))
    assert first.receipt is not None
    assert first.receipt.memory_status is ConversationEventMemoryStatus.FAILED
    service.mark_response_sent(first)
    final = _claimed_receipt(
        repository,
        event_ref="memory-terminal-failure",
    )

    assert final.status is ConversationEventReceiptStatus.RESPONSE_SENT
    assert final.memory_status is ConversationEventMemoryStatus.FAILED
    assert final.memory_error_code == "memory_append_failed"

    duplicate = await service.handle(_command(event_ref="memory-terminal-failure"))
    assert duplicate.should_send is False
    assert memory.append_calls == 1
    assert planner.calls == 1
    assert executor.calls == 1


def test_renderer_bound_fits_receipt_bound() -> None:
    assert MAX_RESPONSE_CHARS <= _MAX_RESPONSE_LENGTH
