# © Artur Czarnecki. All rights reserved.

"""Optional Slack companion lifecycle wrapper for the LKW host."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Callable, cast

from fastapi import FastAPI

from intergrax.applications._shared.fastapi_lifespan import combine_lifespans
from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.conversation_channel import (
    ConversationAttachmentFetcher,
)
from intergrax.integrations.contracts.document_store import DocumentStore
from intergrax.integrations.providers.conversation_channel.slack.config import (
    SlackConversationChannelIntegrationConfig,
)
from intergrax.integrations.providers.conversation_channel.slack.integration import (
    SlackConversationChannelIntegration,
)
from local_workspace_application.host.lifecycle import LocalWorkspaceHostLifecycle
from local_workspace_application.host.settings import LocalWorkspaceBackendSettings
from local_workspace_application.slack_companion.ask_client import (
    SlackAskClientConfig,
    WorkspaceAskHttpClient,
)
from local_workspace_application.slack_companion.authorization import SlackCompanionAuthConfig
from local_workspace_application.slack_companion.dedupe_repository import (
    SlackEventDedupeRepository,
)
from local_workspace_application.slack_companion.selection_store import (
    InMemorySlackWorkspaceSelectionStore,
)
from local_workspace_application.slack_companion.pending_deletion_store import (
    InMemorySlackPendingDeletionStore,
)
from local_workspace_application.slack_companion.workflow import SlackAskWorkflow
from local_workspace_application.conversation.interaction_application_service import (
    ConversationInteractionApplicationService,
)
from local_workspace_application.conversation.interaction_event_receipt import (
    ConversationInteractionEventReceiptRepository,
)
from local_workspace_application.conversation.interaction_executor import (
    ConversationInteractionExecutor,
)
from local_workspace_application.conversation.interaction_planner import (
    ConversationInteractionPlanner,
)
from local_workspace_application.conversation.interaction_response_renderer import (
    ConversationInteractionResponseRenderer,
)
from local_workspace_application.conversation.conversation_thread_memory_service import (
    ConversationThreadMemoryService,
)
from local_workspace_application.workspaces.conversation_context_memory import (
    DocumentStoreThreadMemoryLifecyclePort,
    SessionHistorySnapshotConversationThreadMemoryAdapter,
)
from local_workspace_application.workspaces.conversation_context_models import (
    ConversationThreadMemoryLimitsV1,
    ConversationProductCapability,
)
from local_workspace_application.workspaces.conversation_context_repository import (
    ConversationContextRepository,
)
from local_workspace_application.workspaces.conversation_context_resolution import (
    ConversationContextResolver,
)
from local_workspace_application.workspaces.conversation_workspace_selection_service import (
    ConversationWorkspaceSelectionService,
)
from local_workspace_application.workspaces.document_store_factory import (
    resolve_managed_workspace_document_store,
)

logger = logging.getLogger(__name__)

COMPONENT_NAME = "slack_companion"

# Full env names for LKW Slack Ask companion (LOCAL_WORKSPACE_ + settings keys).
# Shared by operator preflight and .env.example contract tests — not secrets.
SLACK_COMPANION_PRODUCT_ENV_KEYS: tuple[str, ...] = (
    "LOCAL_WORKSPACE_SLACK_COMPANION_ENABLED",
    "LOCAL_WORKSPACE_SLACK_APPROVED_TEAM_ID",
    "LOCAL_WORKSPACE_SLACK_APPROVED_USER_ID",
    "LOCAL_WORKSPACE_SLACK_TENANT_ID",
    "LOCAL_WORKSPACE_SLACK_ACTIVE_WORKSPACE_ID",
    "LOCAL_WORKSPACE_SLACK_ASK_BASE_URL",
    "LOCAL_WORKSPACE_SLACK_ASK_API_KEY",
    "LOCAL_WORKSPACE_SLACK_ASK_TIMEOUT_SECONDS",
    "LOCAL_WORKSPACE_CONVERSATION_THREAD_MEMORY_MAX_MESSAGES",
    "LOCAL_WORKSPACE_CONVERSATION_THREAD_MEMORY_MAX_BYTES",
    "LOCAL_WORKSPACE_CONVERSATION_THREAD_MEMORY_MAX_AGE_SECONDS",
)


@dataclass(frozen=True, slots=True)
class SlackCompanionRuntimeConfig:
    """Validated LKW product mapping for an enabled Slack companion."""

    approved_team_id: str
    approved_user_id: str
    tenant_id: str
    active_workspace_id: str
    ask_base_url: str
    ask_api_key: str | None
    ask_timeout_seconds: float
    conversation_connection_ref: str
    attachment_max_bytes: int
    attachment_max_batch_files: int
    thread_memory_limits: ConversationThreadMemoryLimitsV1


def resolve_slack_companion_runtime_config(
    settings: LocalWorkspaceBackendSettings,
) -> SlackCompanionRuntimeConfig | None:
    """Return validated config when companion is enabled and complete; else ``None``."""
    if not settings.slack_companion_enabled:
        return None

    approved_team_id = settings.slack_approved_team_id.strip()
    approved_user_id = settings.slack_approved_user_id.strip()
    tenant_id = settings.slack_tenant_id.strip()
    active_workspace_id = settings.slack_active_workspace_id.strip()
    ask_base_url = settings.slack_ask_base_url.strip()
    ask_api_key = settings.slack_ask_api_key.strip() or None
    attachment_max_bytes = int(settings.managed_file_max_bytes)
    attachment_max_batch_files = int(settings.managed_file_max_batch_files)

    required = (
        approved_team_id,
        approved_user_id,
        tenant_id,
        active_workspace_id,
        ask_base_url,
    )
    if not all(required):
        return None
    if settings.slack_ask_timeout_seconds <= 0:
        return None
    if attachment_max_bytes < 1 or attachment_max_batch_files < 1:
        return None
    try:
        thread_memory_limits = ConversationThreadMemoryLimitsV1(
            max_messages=settings.conversation_thread_memory_max_messages,
            max_bytes=settings.conversation_thread_memory_max_bytes,
            max_age_seconds=settings.conversation_thread_memory_max_age_seconds,
        )
    except ValueError:
        return None

    return SlackCompanionRuntimeConfig(
        approved_team_id=approved_team_id,
        approved_user_id=approved_user_id,
        tenant_id=tenant_id,
        active_workspace_id=active_workspace_id,
        ask_base_url=ask_base_url,
        ask_api_key=ask_api_key,
        ask_timeout_seconds=float(settings.slack_ask_timeout_seconds),
        conversation_connection_ref=(
            settings.connected_source_slack_connection_ref.strip() or "slack"
        ),
        attachment_max_bytes=attachment_max_bytes,
        attachment_max_batch_files=attachment_max_batch_files,
        thread_memory_limits=thread_memory_limits,
    )


class SlackCompanion:
    """Owns optional SlackConversationChannelIntegration + product workflow."""

    def __init__(
        self,
        *,
        integration: SlackConversationChannelIntegration,
        workflow: SlackAskWorkflow,
    ) -> None:
        self._integration = integration
        self._workflow = workflow
        self._started = False

    @property
    def started(self) -> bool:
        return self._started

    @property
    def integration(self) -> SlackConversationChannelIntegration:
        return self._integration

    @property
    def workflow(self) -> SlackAskWorkflow:
        return self._workflow

    async def start(self) -> None:
        await self._integration.start(self._workflow.handle)
        self._started = True

    async def stop(self) -> None:
        if not self._started:
            return
        try:
            await self._integration.stop()
        finally:
            self._started = False

    def health_detail(self) -> str:
        if not self._started:
            return "not_started"
        status = self._integration.health()
        if isinstance(status, bool):
            return "healthy" if status else "unhealthy"
        return status.detail or ("healthy" if status.healthy else "unhealthy")

    def is_healthy(self) -> bool:
        if not self._started:
            return False
        status = self._integration.health()
        if isinstance(status, bool):
            return status
        return bool(status.healthy)


def build_slack_companion(
    *,
    runtime: SlackCompanionRuntimeConfig,
    document_store: DocumentStore | None = None,
    integration: SlackConversationChannelIntegration | None = None,
    ask_client: WorkspaceAskHttpClient | None = None,
    interaction_application_service: ConversationInteractionApplicationService | None = None,
) -> SlackCompanion:
    """Build a companion from validated product config + platform Slack integration."""
    store = resolve_managed_workspace_document_store(document_store)
    dedupe = SlackEventDedupeRepository(store)
    selections = InMemorySlackWorkspaceSelectionStore()
    pending_deletions = InMemorySlackPendingDeletionStore()
    auth = SlackCompanionAuthConfig(
        approved_team_id=runtime.approved_team_id,
        approved_user_id=runtime.approved_user_id,
        tenant_id=runtime.tenant_id,
        active_workspace_id=runtime.active_workspace_id,
    )
    client = ask_client or WorkspaceAskHttpClient(
        SlackAskClientConfig(
            base_url=runtime.ask_base_url,
            api_key=runtime.ask_api_key,
            timeout_seconds=runtime.ask_timeout_seconds,
        )
    )
    resolved_integration = integration
    if resolved_integration is None:
        platform_config = SlackConversationChannelIntegrationConfig.from_env(enabled=True)
        platform_config.validate_for_runtime()
        resolved_integration = SlackConversationChannelIntegration.from_config(platform_config)

    backend = resolved_integration.backend
    if backend is None:
        raise IntegrationConfigurationError(
            "Slack companion requires a ConversationChannelBackend"
        )
    workflow = SlackAskWorkflow.from_backend(
        backend,
        auth_config=auth,
        dedupe=dedupe,
        ask_client=client,
        selection_store=selections,
        pending_deletion_store=pending_deletions,
        attachment_max_bytes=runtime.attachment_max_bytes,
        attachment_max_batch_files=runtime.attachment_max_batch_files,
        interaction_application_service=interaction_application_service,
        conversation_connection_ref=runtime.conversation_connection_ref,
    )
    return SlackCompanion(integration=resolved_integration, workflow=workflow)


def wire_slack_companion(
    app: FastAPI,
    *,
    settings: LocalWorkspaceBackendSettings,
    host_lifecycle: LocalWorkspaceHostLifecycle | None,
    document_store: DocumentStore | None = None,
    integration_factory: Callable[[], SlackConversationChannelIntegration] | None = None,
    ask_client: WorkspaceAskHttpClient | None = None,
) -> SlackCompanion | None:
    """Register optional ``slack_companion`` readiness and lifespan.

    Never raises into the host bootstrap path for missing/invalid Slack config.
    """
    if host_lifecycle is None:
        return None

    if not settings.slack_companion_enabled:
        host_lifecycle.register_component(
            COMPONENT_NAME,
            enabled=False,
            required=False,
            healthy=True,
            detail="disabled",
        )
        return None

    runtime = resolve_slack_companion_runtime_config(settings)
    if runtime is None:
        host_lifecycle.register_component(
            COMPONENT_NAME,
            enabled=False,
            required=False,
            healthy=True,
            detail="invalid_product_config",
        )
        logger.warning(
            "slack_companion enabled but product config incomplete; companion disabled"
        )
        return None

    try:
        if integration_factory is not None:
            integration = integration_factory()
            interaction_service = (
                _build_conversation_interaction_application_service(
                    app=app,
                    runtime=runtime,
                    integration=integration,
                    document_store=document_store,
                )
                if integration is not None
                else None
            )
            companion = build_slack_companion(
                runtime=runtime,
                document_store=document_store,
                integration=integration,
                ask_client=ask_client,
                interaction_application_service=interaction_service,
            )
        else:
            companion = build_slack_companion(
                runtime=runtime,
                document_store=document_store,
                ask_client=ask_client,
            )
    except Exception as exc:  # noqa: BLE001 — degrade Slack only
        host_lifecycle.register_component(
            COMPONENT_NAME,
            enabled=False,
            required=False,
            healthy=True,
            detail=f"unavailable:{type(exc).__name__}",
        )
        logger.warning(
            "slack_companion unavailable kind=%s; core host continues",
            type(exc).__name__,
        )
        return None

    host_lifecycle.register_component(
        COMPONENT_NAME,
        enabled=True,
        required=False,
        healthy=True,
        detail="configured",
    )
    app.state.lkw_slack_companion = companion
    _apply_slack_companion_lifespan(app, companion, host_lifecycle)
    return companion


class _SlackConnectionAuthorization:
    def __init__(
        self,
        *,
        tenant_id: str,
        connection_ref: str,
        integration: SlackConversationChannelIntegration,
    ) -> None:
        self._tenant_id = tenant_id
        self._connection_ref = connection_ref
        self._integration = integration

    def is_conversation_connection_active_and_tenant_owned(
        self,
        *,
        tenant_id: str,
        conversation_connection_ref: str,
    ) -> bool:
        return (
            tenant_id == self._tenant_id
            and conversation_connection_ref == self._connection_ref
            and self._integration.backend is not None
        )


class _SlackWorkspaceAuthorization:
    def __init__(self, workspace_service: Any, approved_principal_ref: str) -> None:
        self._workspace_service = workspace_service
        self._approved_principal_ref = approved_principal_ref

    def is_workspace_active(self, *, tenant_id: str, workspace_id: str) -> bool:
        return (
            self._workspace_service.get_workspace(
                tenant_id=tenant_id,
                workspace_id=workspace_id,
            )
            is not None
        )

    def may_principal_use_workspace(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        principal_ref: str,
    ) -> bool:
        return principal_ref == self._approved_principal_ref and self.is_workspace_active(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
        )


class _AttachmentResolverBridge:
    def __init__(self) -> None:
        self.service: ConversationInteractionApplicationService | None = None

    def resolve(self, attachment_id: str) -> object | None:
        if self.service is None:
            return None
        return self.service.resolve_attachment(attachment_id)


def _build_conversation_interaction_application_service(
    *,
    app: FastAPI,
    runtime: SlackCompanionRuntimeConfig,
    integration: SlackConversationChannelIntegration,
    document_store: DocumentStore | None,
) -> ConversationInteractionApplicationService:
    workspace_service = app.state.lkw_managed_workspace_service
    store = document_store
    if store is None:
        store = app.state.lkw_managed_workspace_repository.document_store
    context_repository = ConversationContextRepository(store)
    resolver = ConversationContextResolver(
        context_repository,
        connection_port=_SlackConnectionAuthorization(
            tenant_id=runtime.tenant_id,
            connection_ref=runtime.conversation_connection_ref,
            integration=integration,
        ),
        workspace_port=_SlackWorkspaceAuthorization(
            workspace_service,
            runtime.approved_user_id,
        ),
    )
    selection_service = ConversationWorkspaceSelectionService(
        context_repository,
        workspace_service,
        clock=lambda: datetime.now(UTC),
    )
    ask_service = app.state.lkw_ask_service
    planner = ConversationInteractionPlanner(ask_service.llm_adapter)
    bridge = _AttachmentResolverBridge()
    executor = ConversationInteractionExecutor(
        workspace_service=workspace_service,
        workspace_selection_service=selection_service,
        source_candidate_service=getattr(
            app.state,
            "lkw_source_candidate_intake_service",
            None,
        ),
        attachment_intake_service=getattr(
            app.state,
            "lkw_managed_file_intake_service",
            None,
        ),
        trusted_attachment_resolver=bridge.resolve,
        web_url_intake_service=getattr(app.state, "lkw_web_url_intake_service", None),
        ask_service=ask_service,
    )
    def clock() -> datetime:
        return datetime.now(UTC)
    memory_adapter = SessionHistorySnapshotConversationThreadMemoryAdapter(
        port=DocumentStoreThreadMemoryLifecyclePort(store),
    )
    thread_memory = ConversationThreadMemoryService(
        adapter=memory_adapter,
        limits=runtime.thread_memory_limits,
        clock=clock,
    )
    interaction_service = ConversationInteractionApplicationService(
        context_resolver=resolver,
        planner=planner,
        executor=executor,
        renderer=ConversationInteractionResponseRenderer(),
        receipt_repository=ConversationInteractionEventReceiptRepository(store),
        workspace_service=workspace_service,
        source_candidate_service=getattr(
            app.state,
            "lkw_source_candidate_intake_service",
            None,
        ),
        attachment_loader=(
            integration.backend
            if isinstance(integration.backend, ConversationAttachmentFetcher)
            else None
        ),
        personal_allowed_capabilities=frozenset(ConversationProductCapability),
        attachment_max_bytes=runtime.attachment_max_bytes,
        thread_memory_service=thread_memory,
        clock=clock,
    )
    bridge.service = interaction_service
    return interaction_service


def _apply_slack_companion_lifespan(
    app: FastAPI,
    companion: SlackCompanion,
    host_lifecycle: LocalWorkspaceHostLifecycle,
) -> None:
    @asynccontextmanager
    async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
        try:
            await companion.start()
            host_lifecycle.update_component(
                COMPONENT_NAME,
                healthy=companion.is_healthy(),
                detail=companion.health_detail(),
            )
        except Exception as exc:  # noqa: BLE001 — optional component
            host_lifecycle.update_component(
                COMPONENT_NAME,
                healthy=False,
                detail=f"start_failed:{type(exc).__name__}",
            )
            logger.warning(
                "slack_companion start_failed kind=%s",
                type(exc).__name__,
            )
        try:
            yield
        finally:
            try:
                await companion.stop()
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "slack_companion stop_failed kind=%s",
                    type(exc).__name__,
                )
            host_lifecycle.update_component(
                COMPONENT_NAME,
                healthy=True,
                detail="stopped",
            )

    existing = cast(Any, app.router.lifespan_context)
    app.router.lifespan_context = cast(
        Any,
        combine_lifespans(existing, cast(Any, _lifespan)),
    )
