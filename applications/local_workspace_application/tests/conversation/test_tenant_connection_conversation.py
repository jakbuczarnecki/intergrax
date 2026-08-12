# © Artur Czarnecki. All rights reserved.

"""Focused tests for conversational tenant connection journey (PRODUCT-5C)."""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.runtime.vendor_knowledge.tenant_connections import (
    SafeTenantConnectionV1,
    TenantConnectionAdministrativeStatus,
)
from local_workspace_application.conversation.interaction_draft_models import (
    ConversationInteractionDraft,
    TenantConnectionBeginAuthorizationDraftAction,
    TenantConnectionConnectionsListDraftAction,
    TenantConnectionInspectDraftAction,
    TenantConnectionProvidersListDraftAction,
    TenantConnectionReconnectDraftAction,
    TenantConnectionRevokeDraftAction,
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
    TenantConnectionBeginAuthorizationPlannedAction,
    TenantConnectionCompleteManualAuthorizationPlannedAction,
    TenantConnectionConnectionsListPlannedAction,
    TenantConnectionInspectPlannedAction,
    TenantConnectionProvidersListPlannedAction,
    TenantConnectionReconnectPlannedAction,
    TenantConnectionRevokePlannedAction,
    WorkspaceAskPlannedAction,
    WorkspaceReference,
    WorkspaceReferenceKind,
)
from local_workspace_application.conversation.interaction_plan_compiler import (
    ConversationDraftCompilationError,
    compile_interaction_draft,
)
from local_workspace_application.conversation.interaction_planner import (
    PlanRequestValidationError,
    validate_plan_against_request,
)
from local_workspace_application.conversation.interaction_response_renderer import (
    ConversationInteractionResponseRenderer,
)
from local_workspace_application.workspaces.conversation_connection_auth_context_service import (
    ConversationConnectionAuthContextService,
    TenantConnectionConversationConfig,
    parse_manual_credential_payload,
)
from local_workspace_application.workspaces.conversation_context_models import (
    ConversationActivationPolicy,
    ConversationAudienceMode,
    ConversationExecutionContextV1,
    ConversationProductCapability,
    ConversationThreadContextPolicy,
)
from local_workspace_application.workspaces.conversation_context_repository import (
    ConversationContextRepository,
)
from local_workspace_application.workspaces.destructive_action_confirmation import (
    DestructiveActionConfirmationV1,
    HmacDestructiveActionConfirmationCodec,
    tenant_connection_revoke_action_kind,
)
from local_workspace_application.workspaces.tenant_connection_conversation_models import (
    THREAD_MEMORY_CREDENTIAL_REDACTION,
    TenantConnectionPlanningConnectionV1,
    TenantConnectionPlanningProviderV1,
    TenantConnectionPlanningSnapshotV1,
    TenantConnectionPendingManualAuthorizationV1,
)
from local_workspace_application.workspaces.tenant_connection_conversation_resolution import (
    resolve_connection_reference,
    resolve_provider_reference,
)
from local_workspace_application.workspaces.tenant_connection_product_errors import (
    TenantConnectionProductError,
)
from local_workspace_application.workspaces.tenant_connection_product_orchestration import (
    ConnectionAuthBeginResult,
)

pytestmark = pytest.mark.unit

_TENANT = "tenant-a"
_NOW = datetime(2026, 1, 1, tzinfo=UTC)


def _inventory() -> TenantConnectionPlanningSnapshotV1:
    return TenantConnectionPlanningSnapshotV1(
        providers=(
            TenantConnectionPlanningProviderV1(
                provider_id="google_workspace",
                safe_display_name="Google Workspace",
                auth_mode="oauth_delegated",
                qualification="qualified",
            ),
            TenantConnectionPlanningProviderV1(
                provider_id="slack",
                safe_display_name="Slack",
                auth_mode="manual_credential_binding",
                qualification="not_qualified",
            ),
        ),
        connections=(
            TenantConnectionPlanningConnectionV1(
                connection_ref="conn.google.1",
                provider_id="google_workspace",
                safe_display_name="Google Workspace",
                administrative_status="active",
                connected_principal_ref="user@example.com",
                configuration_version=3,
            ),
            TenantConnectionPlanningConnectionV1(
                connection_ref="conn.google.2",
                provider_id="google_workspace",
                safe_display_name="Google Workspace Backup",
                administrative_status="active",
                connected_principal_ref=None,
                configuration_version=1,
            ),
        ),
    )


def _context() -> ConversationExecutionContextV1:
    return ConversationExecutionContextV1(
        tenant_id=_TENANT,
        conversation_context_binding_id="binding-1",
        audience_mode=ConversationAudienceMode.PERSONAL,
        workspace_id="workspace-1",
        principal_ref="principal-1",
        canonical_thread_ref="thread-1",
        activation_policy=ConversationActivationPolicy.ALWAYS,
        thread_context_policy=ConversationThreadContextPolicy.CURRENT_THREAD_BOUNDED,
        allowed_product_capabilities=frozenset(ConversationProductCapability),
    )


def _safe_connection(**overrides: object) -> SafeTenantConnectionV1:
    payload = {
        "connection_ref": "conn.google.1",
        "tenant_id": _TENANT,
        "provider_id": "google_workspace",
        "integration_kind": "collaboration_suite",
        "safe_display_name": "Google Workspace",
        "administrative_status": TenantConnectionAdministrativeStatus.ACTIVE,
        "configuration_version": 3,
        "connected_principal_ref": "user@example.com",
        "created_at": _NOW,
        "updated_at": _NOW,
    }
    payload.update(overrides)
    return SafeTenantConnectionV1(**payload)


class _OrchestrationStub:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    def list_supported_connection_providers(self):
        self.calls.append(("list_supported_connection_providers", (), {}))
        return (
            {
                "provider_id": "google_workspace",
                "integration_kind": "collaboration_suite",
                "auth_mode": "oauth_delegated",
                "safe_display_name": "Google Workspace",
                "supported_scopes_summary": "drive",
                "qualification": "qualified",
            },
        )

    def list_connections(self, **kwargs: object):
        self.calls.append(("list_connections", (), kwargs))
        return (_safe_connection(),)

    def get_connection(self, connection_ref: str):
        self.calls.append(("get_connection", (connection_ref,), {}))
        return _safe_connection(connection_ref=connection_ref)

    def begin_connection_authorization(self, **kwargs: object):
        self.calls.append(("begin_connection_authorization", (), kwargs))
        return ConnectionAuthBeginResult(
            authorization_transaction_ref="auth.google.abc",
            authorization_url="https://oauth.example/authorize",
            expires_at=_NOW,
            required_user_action="redirect",
        )

    def complete_connection_authorization(self, **kwargs: object):
        self.calls.append(("complete_connection_authorization", (), kwargs))
        return SimpleNamespace(
            connection=_safe_connection(),
            disposition="created",
        )

    def reconnect_connection(self, **kwargs: object):
        self.calls.append(("reconnect_connection", (), kwargs))
        return ConnectionAuthBeginResult(
            authorization_transaction_ref="auth.google.reconnect",
            authorization_url="https://oauth.example/reauthorize",
            expires_at=_NOW,
            required_user_action="redirect",
        )

    def revoke_connection(self, **kwargs: object):
        self.calls.append(("revoke_connection", (), kwargs))
        return _safe_connection(administrative_status=TenantConnectionAdministrativeStatus.REVOKED)


class _FactoryStub:
    def __init__(self, service: _OrchestrationStub) -> None:
        self._service = service

    def for_tenant(self, tenant_id: str) -> _OrchestrationStub:
        assert tenant_id == _TENANT
        return self._service


def _executor(
    orchestration: _OrchestrationStub,
    *,
    repository: ConversationContextRepository | None = None,
) -> ConversationInteractionExecutor:
    repo = repository or ConversationContextRepository(InMemoryDocumentStore())
    auth_context = ConversationConnectionAuthContextService(
        context_repository=repo,
        orchestration_factory=_FactoryStub(orchestration),  # type: ignore[arg-type]
        clock=lambda: _NOW,
    )
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


def make_tenant_connection_execution_command(
    *,
    action,
    message_text: str = "connect google",
    inventory: TenantConnectionPlanningSnapshotV1 | None = None,
) -> ConversationInteractionExecutionCommand:
    return ConversationInteractionExecutionCommand(
        tenant_id=_TENANT,
        execution_context=_context(),
        planning_request=ConversationPlanningRequest(
            message_text=message_text,
            available_workspaces=(),
            tenant_connection_inventory=inventory or _inventory(),
        ),
        interaction_plan=ConversationInteractionPlan(
            plan_version="2",
            response_mode="aggregate",
            actions=(action,),
        ),
    )


def test_planned_actions_are_frozen_and_reject_invalid_payload() -> None:
    action = TenantConnectionProvidersListPlannedAction(
        action_id="a1",
        action_type="tenant_connection.providers.list",
    )
    assert action.model_config["frozen"] is True
    with pytest.raises(ValidationError):
        TenantConnectionBeginAuthorizationPlannedAction(
            action_id="a2",
            action_type="tenant_connection.authorization.begin",
            provider_id="",
        )


def test_compiler_maps_provider_and_connection_intents() -> None:
    draft = ConversationInteractionDraft(
        actions=(
            TenantConnectionProvidersListDraftAction(
                action_type="tenant_connection.providers.list",
            ),
            TenantConnectionConnectionsListDraftAction(
                action_type="tenant_connection.connections.list",
            ),
            TenantConnectionBeginAuthorizationDraftAction(
                action_type="tenant_connection.authorization.begin",
                provider_reference="Google Workspace",
            ),
            TenantConnectionInspectDraftAction(
                action_type="tenant_connection.connection.inspect",
                connection_reference="Google Workspace Backup",
            ),
            TenantConnectionReconnectDraftAction(
                action_type="tenant_connection.connection.reconnect",
                connection_reference="Google Workspace Backup",
            ),
            TenantConnectionRevokeDraftAction(
                action_type="tenant_connection.connection.revoke",
                connection_reference="Google Workspace Backup",
            ),
        )
    )
    plan = compile_interaction_draft(
        draft,
        request=ConversationPlanningRequest(
            message_text="connect google",
            available_workspaces=(),
            tenant_connection_inventory=_inventory(),
        ),
    )
    assert isinstance(plan.actions[0], TenantConnectionProvidersListPlannedAction)
    assert isinstance(plan.actions[2], TenantConnectionBeginAuthorizationPlannedAction)
    assert plan.actions[2].provider_id == "google_workspace"
    assert isinstance(plan.actions[3], TenantConnectionInspectPlannedAction)
    assert plan.actions[3].connection_ref == "conn.google.2"


def test_compiler_rejects_missing_inventory() -> None:
    draft = ConversationInteractionDraft(
        actions=(
            TenantConnectionBeginAuthorizationDraftAction(
                action_type="tenant_connection.authorization.begin",
                provider_reference="Google",
            ),
        )
    )
    with pytest.raises(ConversationDraftCompilationError):
        compile_interaction_draft(
            draft,
            request=ConversationPlanningRequest(
                message_text="providers",
                available_workspaces=(),
            ),
        )


def test_resolution_single_zero_and_ambiguous_matches() -> None:
    inventory = _inventory()
    single = resolve_provider_reference(inventory.providers, provider_reference="Google")
    assert single.provider_id == "google_workspace"
    assert single.ambiguous is False

    missing = resolve_provider_reference(inventory.providers, provider_reference="Dropbox")
    assert missing.provider_id is None

    ambiguous = resolve_connection_reference(
        inventory.connections,
        connection_reference="Google",
    )
    assert ambiguous.connection_ref is None
    assert ambiguous.ambiguous is True

    exact = resolve_connection_reference(
        inventory.connections,
        connection_reference="Google Workspace Backup",
    )
    assert exact.connection_ref == "conn.google.2"


def test_provider_list_intent_does_not_compile_to_workspace_ask() -> None:
    draft = ConversationInteractionDraft(
        actions=(
            TenantConnectionProvidersListDraftAction(
                action_type="tenant_connection.providers.list",
            ),
        )
    )
    plan = compile_interaction_draft(
        draft,
        request=ConversationPlanningRequest(
            message_text="jakie integracje mogę podłączyć?",
            available_workspaces=(),
            tenant_connection_inventory=_inventory(),
        ),
    )
    assert len(plan.actions) == 1
    assert isinstance(plan.actions[0], TenantConnectionProvidersListPlannedAction)
    assert not any(isinstance(action, WorkspaceAskPlannedAction) for action in plan.actions)


@pytest.mark.asyncio
async def test_executor_delegates_to_orchestration_without_provider_adapters() -> None:
    orchestration = _OrchestrationStub()
    executor = _executor(orchestration)
    result = await executor.execute(
        make_tenant_connection_execution_command(
            action=TenantConnectionProvidersListPlannedAction(
                action_id="a1",
                action_type="tenant_connection.providers.list",
            )
        )
    )
    assert result.status is ConversationInteractionOverallStatus.COMPLETED
    assert orchestration.calls[0][0] == "list_supported_connection_providers"


@pytest.mark.asyncio
async def test_oauth_begin_renderer_exposes_url_without_internal_refs() -> None:
    orchestration = _OrchestrationStub()
    executor = _executor(orchestration)
    result = await executor.execute(
        make_tenant_connection_execution_command(
            action=TenantConnectionBeginAuthorizationPlannedAction(
                action_id="a1",
                action_type="tenant_connection.authorization.begin",
                provider_id="google_workspace",
            )
        )
    )
    artifact = result.action_results[0].artifact
    assert artifact is not None
    assert artifact.data["authorization_url"] == "https://oauth.example/authorize"
    assert "authorization_transaction_ref" not in artifact.data
    rendered = ConversationInteractionResponseRenderer().render(result)
    assert "https://oauth.example/authorize" in rendered
    assert "auth.google" not in rendered


@pytest.mark.asyncio
async def test_manual_slack_credentials_are_redacted_from_thread_memory() -> None:
    orchestration = _OrchestrationStub()

    def begin(**kwargs: object):
        return ConnectionAuthBeginResult(
            authorization_transaction_ref="auth.slack.abc",
            authorization_url=None,
            expires_at=_NOW,
            required_user_action="present_manual_instructions",
            manual_instructions="Send JSON with app_token and bot_token.",
        )

    orchestration.begin_connection_authorization = begin  # type: ignore[method-assign]
    repo = ConversationContextRepository(InMemoryDocumentStore())
    executor = _executor(orchestration, repository=repo)
    await executor.execute(
        make_tenant_connection_execution_command(
            action=TenantConnectionBeginAuthorizationPlannedAction(
                action_id="a1",
                action_type="tenant_connection.authorization.begin",
                provider_id="slack",
            ),
            inventory=TenantConnectionPlanningSnapshotV1(
                providers=_inventory().providers,
                connections=(),
                pending_manual_authorization=None,
            ),
        )
    )
    credential_message = '{"app_token":"xapp-1","bot_token":"xoxb-1"}'
    result = await executor.execute(
        make_tenant_connection_execution_command(
            action=TenantConnectionCompleteManualAuthorizationPlannedAction(
                action_id="a2",
                action_type="tenant_connection.authorization.complete_manual",
            ),
            message_text=credential_message,
            inventory=TenantConnectionPlanningSnapshotV1(
                providers=_inventory().providers,
                connections=(),
                pending_manual_authorization=TenantConnectionPendingManualAuthorizationV1(
                    authorization_transaction_ref="auth.slack.abc",
                    provider_id="slack",
                ),
            ),
        )
    )
    assert result.thread_memory_user_text == THREAD_MEMORY_CREDENTIAL_REDACTION
    rendered = ConversationInteractionResponseRenderer().render(result)
    assert "xapp-1" not in rendered
    assert "xoxb-1" not in rendered
    assert orchestration.calls[-1][0] == "complete_connection_authorization"


def test_manual_credential_parser_rejects_invalid_payload() -> None:
    with pytest.raises(ValueError, match="credential_binding_invalid"):
        parse_manual_credential_payload("not-json")


@pytest.mark.asyncio
async def test_reconnect_preserves_connection_ref_and_revoked_fails() -> None:
    orchestration = _OrchestrationStub()
    executor = _executor(orchestration)
    result = await executor.execute(
        make_tenant_connection_execution_command(
            action=TenantConnectionReconnectPlannedAction(
                action_id="a1",
                action_type="tenant_connection.connection.reconnect",
                connection_ref="conn.google.1",
            )
        )
    )
    assert result.status is ConversationInteractionOverallStatus.COMPLETED
    reconnect_calls = [
        call for call in orchestration.calls if call[0] == "reconnect_connection"
    ]
    assert reconnect_calls[-1] == (
        "reconnect_connection",
        (),
        {
            "connection_ref": "conn.google.1",
            "redirect_uri": "https://app.example.com/oauth/callback",
        },
    )

    def reconnect_revoked(**kwargs: object):
        raise TenantConnectionProductError("connection_revoked")

    orchestration.reconnect_connection = reconnect_revoked  # type: ignore[method-assign]
    failed = await executor.execute(
        make_tenant_connection_execution_command(
            action=TenantConnectionReconnectPlannedAction(
                action_id="a2",
                action_type="tenant_connection.connection.reconnect",
                connection_ref="conn.google.1",
            )
        )
    )
    assert failed.action_results[0].status is ConversationActionExecutionStatus.FAILED


@pytest.mark.asyncio
async def test_revoke_requires_confirmation_then_calls_orchestration() -> None:
    orchestration = _OrchestrationStub()
    executor = _executor(orchestration)
    pending = await executor.execute(
        make_tenant_connection_execution_command(
            action=TenantConnectionRevokePlannedAction(
                action_id="a1",
                action_type="tenant_connection.connection.revoke",
                connection_ref="conn.google.1",
            )
        )
    )
    token = pending.action_results[0].artifact.data["confirmation_token"]  # type: ignore[index]
    confirmed = await executor.execute(
        make_tenant_connection_execution_command(
            action=TenantConnectionRevokePlannedAction(
                action_id="a2",
                action_type="tenant_connection.connection.revoke",
                connection_ref="conn.google.1",
                confirmation_token=token,
            )
        )
    )
    assert confirmed.status is ConversationInteractionOverallStatus.COMPLETED
    assert orchestration.calls[-1][0] == "revoke_connection"


def test_planner_validation_requires_inventory_for_connection_actions() -> None:
    plan = ConversationInteractionPlan(
        plan_version="2",
        response_mode="aggregate",
        actions=(
            TenantConnectionConnectionsListPlannedAction(
                action_id="a1",
                action_type="tenant_connection.connections.list",
            ),
        ),
    )
    with pytest.raises(PlanRequestValidationError):
        validate_plan_against_request(
            plan,
            ConversationPlanningRequest(
                message_text="list connections",
                available_workspaces=(),
            ),
        )


def test_workspace_ask_plan_still_validates_without_tenant_inventory() -> None:
    plan = ConversationInteractionPlan(
        plan_version="2",
        response_mode="aggregate",
        actions=(
            WorkspaceAskPlannedAction(
                action_id="a1",
                action_type="workspace.ask",
                workspace=WorkspaceReference(kind=WorkspaceReferenceKind.active, value=None),
                question="hello",
            ),
        ),
    )
    validate_plan_against_request(
        plan,
        ConversationPlanningRequest(
            message_text="hello",
            available_workspaces=(),
        ),
    )


def test_task_owned_production_files_have_zero_dynamic_wiring() -> None:
    import pathlib
    import re

    root = pathlib.Path("applications/local_workspace_application/workspaces")
    owned = [
        root / "tenant_connection_conversation_models.py",
        root / "tenant_connection_conversation_resolution.py",
        root / "conversation_connection_auth_context_service.py",
    ]
    pattern = re.compile(r"\b(getattr|setattr|hasattr)\s*\(")
    offenders: list[str] = []
    for path in owned:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        if pattern.search(text):
            offenders.append(str(path))
    assert offenders == []
