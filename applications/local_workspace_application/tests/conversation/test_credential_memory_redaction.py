# © Artur Czarnecki. All rights reserved.

"""Security regressions for manual credential thread-memory redaction (PRODUCT-5C-R1)."""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.runtime.vendor_knowledge.tenant_connections import (
    SafeTenantConnectionV1,
    TenantConnectionAdministrativeStatus,
)
from local_workspace_application.conversation.interaction_application_service import (
    ConversationInteractionApplicationCommand,
    ConversationInteractionApplicationService,
)
from local_workspace_application.conversation.interaction_event_receipt import (
    ConversationEventMemoryStatus,
    ConversationEventReceiptError,
    ConversationInteractionEventReceiptRepository,
)
from local_workspace_application.conversation.interaction_execution_models import (
    ConversationInteractionOverallStatus,
)
from local_workspace_application.conversation.interaction_executor import (
    ConversationInteractionExecutor,
)
from local_workspace_application.conversation.interaction_models import (
    ConversationInteractionPlan,
    ConversationPlanningRequest,
    TenantConnectionBeginAuthorizationPlannedAction,
    TenantConnectionCompleteManualAuthorizationPlannedAction,
)
from local_workspace_application.conversation.interaction_response_renderer import (
    ConversationInteractionResponseRenderer,
)
from local_workspace_application.tests.conversation.test_interaction_application_service import (
    _CountingThreadMemory,
    _Executor,
    _MarkerFailureReceiptRepository,
    _Planner,
    _Renderer,
    _Resolver,
    _WorkspaceService,
    _command,
)
from local_workspace_application.workspaces.conversation_connection_auth_context_service import (
    ConversationConnectionAuthContextService,
    TenantConnectionConversationConfig,
)
from local_workspace_application.workspaces.conversation_context_models import (
    ConversationExecutionContextV1,
    ConversationProductCapability,
)
from local_workspace_application.workspaces.conversation_context_repository import (
    ConversationContextRepository,
)
from local_workspace_application.workspaces.destructive_action_confirmation import (
    HmacDestructiveActionConfirmationCodec,
)
from local_workspace_application.workspaces.tenant_connection_conversation_models import (
    THREAD_MEMORY_CREDENTIAL_REDACTION,
)
from local_workspace_application.workspaces.tenant_connection_product_errors import (
    TenantConnectionProductError,
)
from local_workspace_application.workspaces.tenant_connection_product_orchestration import (
    ConnectionAuthBeginResult,
)
from local_workspace_application.workspaces.conversation_context_execution import (
    build_conversation_execution_context,
)

pytestmark = pytest.mark.unit

_TENANT = "tenant-a"
_NOW = datetime(2026, 1, 1, tzinfo=UTC)
_CREDENTIAL_MESSAGE = '{"app_token":"xapp-secret","bot_token":"xoxb-secret"}'
_CLOCK = lambda: datetime(2026, 8, 5, 8, 0, tzinfo=UTC)


def _safe_connection() -> SafeTenantConnectionV1:
    return SafeTenantConnectionV1(
        connection_ref="conn.slack.1",
        tenant_id=_TENANT,
        provider_id="slack",
        integration_kind="collaboration_suite",
        safe_display_name="Slack",
        administrative_status=TenantConnectionAdministrativeStatus.ACTIVE,
        configuration_version=1,
        connected_principal_ref=None,
        created_at=_NOW,
        updated_at=_NOW,
    )


class _OrchestrationStub:
    def __init__(self) -> None:
        self.fail_complete = False

    def list_supported_connection_providers(self):
        return ()

    def list_connections(self, **kwargs: object):
        return ()

    def begin_connection_authorization(self, **kwargs: object):
        return ConnectionAuthBeginResult(
            authorization_transaction_ref="auth.slack.abc",
            authorization_url=None,
            expires_at=_NOW,
            required_user_action="present_manual_instructions",
            manual_instructions="Send JSON with app_token and bot_token.",
        )

    def complete_connection_authorization(self, **kwargs: object):
        if self.fail_complete:
            raise TenantConnectionProductError("credential_binding_invalid")
        return SimpleNamespace(
            connection=_safe_connection(),
            disposition="created",
        )


class _CredentialPlanner:
    async def plan(self, request: ConversationPlanningRequest, **_: object):
        return ConversationInteractionPlan(
            plan_version="2",
            response_mode="aggregate",
            actions=(
                TenantConnectionCompleteManualAuthorizationPlannedAction(
                    action_id="complete-1",
                    action_type="tenant_connection.authorization.complete_manual",
                ),
            ),
        )


class _OAuthBeginPlanner:
    async def plan(self, request: ConversationPlanningRequest, **_: object):
        return ConversationInteractionPlan(
            plan_version="2",
            response_mode="aggregate",
            actions=(
                TenantConnectionBeginAuthorizationPlannedAction(
                    action_id="begin-1",
                    action_type="tenant_connection.authorization.begin",
                    provider_id="slack",
                ),
            ),
        )


def _execution_context() -> ConversationExecutionContextV1:
    resolved = _Resolver().resolve(
        tenant_id=_TENANT,
        ingress=_command().ingress,
    )
    return build_conversation_execution_context(
        resolved=resolved,
        personal_allowed_capabilities=frozenset(ConversationProductCapability),
    )


def _auth_context(
    orchestration: _OrchestrationStub,
) -> ConversationConnectionAuthContextService:
    return ConversationConnectionAuthContextService(
        context_repository=ConversationContextRepository(InMemoryDocumentStore()),
        orchestration_factory=SimpleNamespace(for_tenant=lambda _: orchestration),  # type: ignore[arg-type]
        clock=lambda: _NOW,
    )


def _executor(
    orchestration: _OrchestrationStub,
    auth_context: ConversationConnectionAuthContextService,
) -> ConversationInteractionExecutor:
    codec = HmacDestructiveActionConfirmationCodec(secret=b"test-secret", clock=lambda: _NOW)
    return ConversationInteractionExecutor(
        workspace_service=Mock(),
        workspace_selection_service=Mock(),
        connection_auth_context_service=auth_context,
        tenant_connection_config=TenantConnectionConversationConfig(
            oauth_redirect_uri="https://app.example.com/oauth/callback",
        ),
        destructive_confirmation_codec=codec,
        clock=lambda: _NOW,
    )


def _credential_service(
    *,
    orchestration: _OrchestrationStub,
    memory: _CountingThreadMemory,
    repository: ConversationInteractionEventReceiptRepository,
    seed_pending: bool = True,
) -> ConversationInteractionApplicationService:
    auth_context = _auth_context(orchestration)
    if seed_pending:
        auth_context.record_pending_authorization(
            context=_execution_context(),
            authorization_transaction_ref="auth.slack.abc",
            provider_id="slack",
            required_user_action="present_manual_instructions",
        )
    return ConversationInteractionApplicationService(
        context_resolver=_Resolver(),  # type: ignore[arg-type]
        planner=_CredentialPlanner(),  # type: ignore[arg-type]
        executor=_executor(orchestration, auth_context),
        renderer=ConversationInteractionResponseRenderer(),
        receipt_repository=repository,
        workspace_service=_WorkspaceService(),
        personal_allowed_capabilities=frozenset(ConversationProductCapability),
        thread_memory_service=memory,  # type: ignore[arg-type]
        connection_auth_context_service=auth_context,
        clock=_CLOCK,
    )


def _credential_command(*, event_ref: str = "credential-event-1") -> ConversationInteractionApplicationCommand:
    command = _command(event_ref=event_ref)
    return command.model_copy(update={"message_text": _CREDENTIAL_MESSAGE, "attachments": ()})


def _assert_no_credential_material(payload: object) -> None:
    serialized = str(payload)
    assert "xapp-secret" not in serialized
    assert "xoxb-secret" not in serialized
    assert "app_token" not in serialized
    assert "bot_token" not in serialized


def _user_turn_text(turns: tuple[object, ...]) -> str:
    for turn in turns:
        if turn.role == "user":  # type: ignore[attr-defined]
            return turn.text  # type: ignore[attr-defined]
    raise AssertionError("user turn not found")


@pytest.mark.asyncio
async def test_application_service_successful_credential_submission_redacts_memory() -> None:
    store = InMemoryDocumentStore()
    memory = _CountingThreadMemory(store)
    orchestration = _OrchestrationStub()
    service = _credential_service(
        orchestration=orchestration,
        memory=memory,
        repository=ConversationInteractionEventReceiptRepository(store),
    )
    result = await service.handle(_credential_command())
    assert result.execution_result.status is ConversationInteractionOverallStatus.COMPLETED
    turns = memory.persisted_turns()
    assert _user_turn_text(turns) == THREAD_MEMORY_CREDENTIAL_REDACTION
    _assert_no_credential_material(turns)
    assert result.receipt is not None
    assert result.receipt.safe_user_memory_text == THREAD_MEMORY_CREDENTIAL_REDACTION
    _assert_no_credential_material(result.receipt.model_dump())


@pytest.mark.asyncio
async def test_application_service_malformed_credential_redacts_memory_on_failure() -> None:
    store = InMemoryDocumentStore()
    memory = _CountingThreadMemory(store)
    orchestration = _OrchestrationStub()
    service = _credential_service(
        orchestration=orchestration,
        memory=memory,
        repository=ConversationInteractionEventReceiptRepository(store),
    )
    command = _credential_command().model_copy(update={"message_text": "not-json"})
    result = await service.handle(command)
    assert result.execution_result.status is ConversationInteractionOverallStatus.FAILED
    turns = memory.persisted_turns()
    assert _user_turn_text(turns) == THREAD_MEMORY_CREDENTIAL_REDACTION
    assert "not-json" not in _user_turn_text(turns)
    _assert_no_credential_material(result.receipt)


@pytest.mark.asyncio
async def test_application_service_orchestration_failure_redacts_memory() -> None:
    store = InMemoryDocumentStore()
    memory = _CountingThreadMemory(store)
    orchestration = _OrchestrationStub()
    orchestration.fail_complete = True
    service = _credential_service(
        orchestration=orchestration,
        memory=memory,
        repository=ConversationInteractionEventReceiptRepository(store),
    )
    result = await service.handle(_credential_command())
    assert result.execution_result.status is ConversationInteractionOverallStatus.FAILED
    turns = memory.persisted_turns()
    assert _user_turn_text(turns) == THREAD_MEMORY_CREDENTIAL_REDACTION
    _assert_no_credential_material(turns)


@pytest.mark.asyncio
async def test_duplicate_credential_recovery_redacts_memory() -> None:
    store = InMemoryDocumentStore()
    memory = _CountingThreadMemory(store)
    orchestration = _OrchestrationStub()
    repository = _MarkerFailureReceiptRepository(store, completion_failures=1)
    service = _credential_service(
        orchestration=orchestration,
        memory=memory,
        repository=repository,
    )
    first = await service.handle(_credential_command(event_ref="credential-dup-1"))
    assert first.receipt is not None
    assert first.receipt.memory_status is ConversationEventMemoryStatus.PENDING
    assert first.receipt.safe_user_memory_text == THREAD_MEMORY_CREDENTIAL_REDACTION
    service.mark_response_sent(first)

    duplicate = await service.handle(_credential_command(event_ref="credential-dup-1"))
    assert duplicate.receipt is not None
    assert duplicate.receipt.memory_status is ConversationEventMemoryStatus.COMPLETED
    turns = memory.persisted_turns()
    assert _user_turn_text(turns) == THREAD_MEMORY_CREDENTIAL_REDACTION
    _assert_no_credential_material(turns)


@pytest.mark.asyncio
async def test_memory_append_retry_retains_redacted_representation() -> None:
    store = InMemoryDocumentStore()
    memory = _CountingThreadMemory(store, failures_remaining=1)
    orchestration = _OrchestrationStub()
    repository = _MarkerFailureReceiptRepository(store, failure_marker_failures=1)
    service = _credential_service(
        orchestration=orchestration,
        memory=memory,
        repository=repository,
    )
    first = await service.handle(_credential_command(event_ref="credential-retry-1"))
    assert first.receipt is not None
    assert first.receipt.memory_status is ConversationEventMemoryStatus.PENDING
    service.mark_response_sent(first)

    duplicate = await service.handle(_credential_command(event_ref="credential-retry-1"))
    assert duplicate.receipt is not None
    assert duplicate.receipt.memory_status is ConversationEventMemoryStatus.COMPLETED
    turns = memory.persisted_turns()
    assert _user_turn_text(turns) == THREAD_MEMORY_CREDENTIAL_REDACTION
    assert memory.append_calls == 2


@pytest.mark.asyncio
async def test_ordinary_message_text_still_stored_normally() -> None:
    store = InMemoryDocumentStore()
    memory = _CountingThreadMemory(store)
    service = ConversationInteractionApplicationService(
        context_resolver=_Resolver(),  # type: ignore[arg-type]
        planner=_Planner(),  # type: ignore[arg-type]
        executor=_Executor(),  # type: ignore[arg-type]
        renderer=_Renderer(),  # type: ignore[arg-type]
        receipt_repository=ConversationInteractionEventReceiptRepository(store),
        workspace_service=_WorkspaceService(),
        personal_allowed_capabilities=frozenset(ConversationProductCapability),
        thread_memory_service=memory,  # type: ignore[arg-type]
        clock=_CLOCK,
    )
    message = "list workspaces"
    command = _command(event_ref="ordinary-1").model_copy(
        update={"message_text": message, "attachments": ()}
    )
    await service.handle(command)
    turns = memory.persisted_turns()
    assert _user_turn_text(turns) == message


@pytest.mark.asyncio
async def test_oauth_begin_message_text_still_stored_normally() -> None:
    store = InMemoryDocumentStore()
    memory = _CountingThreadMemory(store)
    orchestration = _OrchestrationStub()
    auth_context = _auth_context(orchestration)
    service = ConversationInteractionApplicationService(
        context_resolver=_Resolver(),  # type: ignore[arg-type]
        planner=_OAuthBeginPlanner(),  # type: ignore[arg-type]
        executor=_executor(orchestration, auth_context),
        renderer=ConversationInteractionResponseRenderer(),
        receipt_repository=ConversationInteractionEventReceiptRepository(store),
        workspace_service=_WorkspaceService(),
        personal_allowed_capabilities=frozenset(ConversationProductCapability),
        thread_memory_service=memory,  # type: ignore[arg-type]
        connection_auth_context_service=auth_context,
        clock=_CLOCK,
    )
    message = "connect slack"
    command = _command(event_ref="oauth-begin-1").model_copy(
        update={"message_text": message, "attachments": ()}
    )
    await service.handle(command)
    turns = memory.persisted_turns()
    assert _user_turn_text(turns) == message


def test_receipt_rejects_pending_memory_without_safe_user_memory_text() -> None:
    repository = ConversationInteractionEventReceiptRepository(InMemoryDocumentStore())
    processing = repository.claim(
        tenant_id="tenant-a",
        conversation_connection_ref="slack",
        provider_event_ref="event-1",
        execution_id="execution-1",
    ).receipt
    with pytest.raises(ConversationEventReceiptError) as error:
        repository.mark_response_pending(
            receipt=processing,
            response="safe response",
            memory_required=True,
        )
    assert error.value.error_code == "conversation_receipt_memory_text_empty"
