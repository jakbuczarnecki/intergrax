# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from intergrax.integrations.contracts.base import HealthStatus
from intergrax.integrations.contracts.conversation_channel import (
    ConversationDeliveryReceipt,
    ConversationEventHandler,
    OutboundConversationMessage,
)
from intergrax.integrations.providers.conversation_channel.slack.integration import (
    SlackConversationChannelIntegration,
)
from local_workspace_application.host.factory import create_local_workspace_backend_app
from local_workspace_application.tests.lkw_ac3_projection import build_lkw_test_registry_projection
from local_workspace_application.host.lifecycle import HostLifecycleState, LocalWorkspaceHostLifecycle
from local_workspace_application.host.settings import LocalWorkspaceBackendSettings
from local_workspace_application.slack_companion.companion import (
    COMPONENT_NAME,
    wire_slack_companion,
)

pytestmark = pytest.mark.unit

_PREFIX = "/v1/local_workspace"


class _FakeBackend:
    def __init__(self) -> None:
        self.handler: ConversationEventHandler | None = None
        self.started = False
        self.stopped = False

    async def start(self, handler: ConversationEventHandler) -> None:
        self.handler = handler
        self.started = True

    async def stop(self) -> None:
        self.stopped = True
        self.started = False

    async def send(self, message: OutboundConversationMessage) -> ConversationDeliveryReceipt:
        return ConversationDeliveryReceipt(
            message_id="m1",
            address=message.address,
            delivered_at=datetime.now(timezone.utc),
        )

    def health(self) -> HealthStatus:
        return HealthStatus(slug="slack", healthy=True, detail="fake-ok")


def _component(lifecycle: LocalWorkspaceHostLifecycle, name: str):
    for item in lifecycle.component_health():
        if item.name == name:
            return item
    raise AssertionError(f"missing component {name}")


def test_slack_disabled_does_not_block_host() -> None:
    settings = LocalWorkspaceBackendSettings(
        include_mcp=False,
        include_scheduler=False,
        slack_companion_enabled=False,
    )
    app = create_local_workspace_backend_app(registry_projection=build_lkw_test_registry_projection(settings), settings=settings)
    lifecycle = app.state.lkw_host_lifecycle
    with TestClient(app) as client:
        assert lifecycle.state is HostLifecycleState.READY
        response = client.get(f"{_PREFIX}/readiness")
        assert response.status_code == 200
        body = response.json()
        assert body["ready"] is True
        slack = next(c for c in body["components"] if c["name"] == COMPONENT_NAME)
        assert slack["enabled"] is False
        assert slack["required"] is False


def test_invalid_slack_config_does_not_block_core_readiness() -> None:
    settings = LocalWorkspaceBackendSettings(
        include_mcp=False,
        include_scheduler=False,
        slack_companion_enabled=True,
        slack_approved_team_id="",
        slack_approved_user_id="U1",
        slack_tenant_id="t",
        slack_active_workspace_id="ws",
        slack_ask_base_url="http://localhost:8020",
    )
    app = create_local_workspace_backend_app(registry_projection=build_lkw_test_registry_projection(settings), settings=settings)
    lifecycle = app.state.lkw_host_lifecycle
    with TestClient(app) as client:
        assert lifecycle.is_ready() is True
        response = client.get(f"{_PREFIX}/readiness")
        assert response.json()["ready"] is True
        slack = _component(lifecycle, COMPONENT_NAME)
        assert slack.enabled is False
        assert slack.required is False
        assert "invalid_product_config" in slack.detail


def test_enabled_fake_slack_starts_and_shutdown_stops() -> None:
    backend = _FakeBackend()
    settings = LocalWorkspaceBackendSettings(
        include_mcp=False,
        include_scheduler=False,
        slack_companion_enabled=False,
    )
    app = create_local_workspace_backend_app(registry_projection=build_lkw_test_registry_projection(settings), settings=settings)
    lifecycle = app.state.lkw_host_lifecycle
    enabled_settings = LocalWorkspaceBackendSettings(
        include_mcp=False,
        include_scheduler=False,
        slack_companion_enabled=True,
        slack_approved_team_id="T1",
        slack_approved_user_id="U1",
        slack_tenant_id="tenant",
        slack_active_workspace_id="ws",
        slack_ask_base_url="http://127.0.0.1:8020",
    )
    wire_slack_companion(
        app,
        settings=enabled_settings,
        host_lifecycle=lifecycle,
        integration_factory=lambda: SlackConversationChannelIntegration.from_backend(
            backend,
            enabled=True,
        ),
    )
    with TestClient(app):
        assert backend.started is True
        assert backend.handler is not None
        slack = _component(lifecycle, COMPONENT_NAME)
        assert slack.enabled is True
        assert slack.required is False
        assert lifecycle.is_ready() is True
    assert backend.stopped is True


def test_http_mcp_readiness_independent_of_slack_health() -> None:
    lifecycle = LocalWorkspaceHostLifecycle()
    lifecycle.set_executor_available(True)
    lifecycle.register_component("runtime", enabled=True, required=True, healthy=True)
    lifecycle.register_component("http", enabled=True, required=True, healthy=True)
    lifecycle.register_component("mcp", enabled=True, required=False, healthy=True)
    lifecycle.register_component(
        COMPONENT_NAME,
        enabled=True,
        required=False,
        healthy=False,
        detail="unhealthy",
    )
    lifecycle.transition_to_ready()
    assert lifecycle.is_ready() is True
    snap = lifecycle.readiness_snapshot()
    assert snap.ready is True
