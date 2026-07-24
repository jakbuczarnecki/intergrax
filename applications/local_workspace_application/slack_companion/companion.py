# © Artur Czarnecki. All rights reserved.

"""Optional Slack companion lifecycle wrapper for the LKW host."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Callable, cast

from fastapi import FastAPI

from intergrax.applications._shared.fastapi_lifespan import combine_lifespans
from intergrax.integrations.contracts.base import IntegrationConfigurationError
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
from local_workspace_application.slack_companion.workflow import SlackAskWorkflow
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

    return SlackCompanionRuntimeConfig(
        approved_team_id=approved_team_id,
        approved_user_id=approved_user_id,
        tenant_id=tenant_id,
        active_workspace_id=active_workspace_id,
        ask_base_url=ask_base_url,
        ask_api_key=ask_api_key,
        ask_timeout_seconds=float(settings.slack_ask_timeout_seconds),
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
) -> SlackCompanion:
    """Build a companion from validated product config + platform Slack integration."""
    store = resolve_managed_workspace_document_store(document_store)
    dedupe = SlackEventDedupeRepository(store)
    selections = InMemorySlackWorkspaceSelectionStore()
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
            companion = build_slack_companion(
                runtime=runtime,
                document_store=document_store,
                integration=integration,
                ask_client=ask_client,
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
        combine_lifespans(existing, _lifespan),
    )
