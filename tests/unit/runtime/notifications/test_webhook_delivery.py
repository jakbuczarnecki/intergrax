# © Artur Czarnecki. All rights reserved.

import httpx
import pytest

from intergrax.integrations.providers.notification_channel.log.integration import LogNotificationChannelIntegration
from intergrax.integrations.providers.notification_channel.slack.integration import SlackNotificationChannelIntegration
from intergrax.integrations.providers.notification_channel.teams.integration import TeamsNotificationChannelIntegration
from intergrax.runtime.notifications.models import NotificationMessage
from intergrax.runtime.notifications.adapters.logging_adapter import LoggingNotificationAdapter
from intergrax.runtime.notifications.adapters.webhook_adapter import WebhookNotificationAdapter
from intergrax.runtime.notifications.deliveries.http_webhook_delivery import HttpWebhookDelivery
from intergrax.runtime.notifications.factory import (
    NotificationBackend,
    create_notification_adapter,
    resolve_notification_adapter,
    resolve_notification_settings,
)
from intergrax.runtime.notifications.formatters import (
    GenericJsonPayloadFormatter,
    SlackPayloadFormatter,
    TeamsPayloadFormatter,
)


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.gate
async def test_logging_notification_adapter(caplog):
    adapter = LoggingNotificationAdapter()
    await adapter.notify(
        NotificationMessage(
            channel="log",
            subject="Progress",
            body="50% complete",
            task_id="task_1",
            tenant_id="t1",
        )
    )


@pytest.mark.unit
@pytest.mark.gate
def test_resolve_notification_adapter_channels():
    assert isinstance(resolve_notification_adapter("log"), LogNotificationChannelIntegration)
    settings = resolve_notification_settings(
        backend="slack",
        slack_webhook_url="https://hooks.example.test/slack",
    )
    slack = create_notification_adapter(settings)
    assert isinstance(slack, SlackNotificationChannelIntegration)
    assert isinstance(slack._require_client(), WebhookNotificationAdapter)
    settings = resolve_notification_settings(
        backend="teams",
        teams_webhook_url="https://hooks.example.test/teams",
    )
    teams = create_notification_adapter(settings)
    assert isinstance(teams, TeamsNotificationChannelIntegration)
    assert isinstance(teams._require_client(), WebhookNotificationAdapter)


@pytest.mark.unit
@pytest.mark.gate
def test_payload_formatters_include_task_context():
    message = NotificationMessage(
        channel="slack",
        subject="Paused",
        body="awaiting approval",
        task_id="task_1",
        tenant_id="t1",
        metadata={"resume_token": "tok_abc"},
    )
    slack = SlackPayloadFormatter().format(message)
    assert "tok_abc" in slack["text"]
    teams = TeamsPayloadFormatter().format(message)
    assert teams["title"] == "Paused"
    generic = GenericJsonPayloadFormatter().format(message)
    assert generic["metadata"]["resume_token"] == "tok_abc"


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.gate
async def test_webhook_adapter_delivers_via_injected_transport():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        captured["url"] = str(request.url)
        captured["json"] = json.loads(request.content.decode())
        return httpx.Response(200, json={"ok": True})

    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport)
    delivery = HttpWebhookDelivery(client=client)
    adapter = WebhookNotificationAdapter(
        webhook_url="https://hooks.example.test/notify",
        formatter=SlackPayloadFormatter(),
        delivery=delivery,
        channel="slack",
        fail_silently=False,
    )

    await adapter.notify(
        NotificationMessage(
            channel="slack",
            subject="Task paused",
            body="Review required",
            task_id="task_1",
            tenant_id="t1",
        )
    )

    assert captured["url"] == "https://hooks.example.test/notify"
    assert "Task paused" in captured["json"]["text"]


@pytest.mark.unit
@pytest.mark.gate
def test_create_notification_adapter_falls_back_to_log_without_url(monkeypatch):
    monkeypatch.delenv("INTERGRAX_SLACK_WEBHOOK_URL", raising=False)
    settings = resolve_notification_settings(backend="slack", slack_webhook_url="")
    adapter = create_notification_adapter(settings)
    assert isinstance(adapter, SlackNotificationChannelIntegration)
    assert isinstance(adapter._require_client(), LoggingNotificationAdapter)


@pytest.mark.unit
@pytest.mark.gate
def test_create_notification_adapter_webhook_backend(monkeypatch):
    settings = resolve_notification_settings(
        backend="webhook",
        webhook_url="https://hooks.example.test/generic",
    )
    adapter = create_notification_adapter(settings)
    assert isinstance(adapter, WebhookNotificationAdapter)
    assert adapter.webhook_url == "https://hooks.example.test/generic"


@pytest.mark.unit
@pytest.mark.gate
def test_notification_backend_enum_values():
    assert NotificationBackend.SLACK.value == "slack"
    assert NotificationBackend.WEBHOOK.value == "webhook"
