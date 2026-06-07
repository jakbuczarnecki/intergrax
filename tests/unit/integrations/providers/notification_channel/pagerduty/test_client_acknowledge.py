# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from unittest.mock import Mock

import pytest

from intergrax.integrations.providers.notification_channel.pagerduty.client import PagerDutyEventsClient
from intergrax.integrations.providers.notification_channel.pagerduty.config import PagerDutyIntegrationConfig

pytestmark = pytest.mark.gate


def test_pagerduty_client_acknowledge_incident_posts_events_api() -> None:
    http = Mock()
    response = Mock()
    response.raise_for_status.return_value = None
    http.post.return_value = response
    client = PagerDutyEventsClient(
        PagerDutyIntegrationConfig(routing_key="routing-key"),
        http_client=http,
    )

    client.acknowledge_incident(dedup_key="dedup-123", note="on it")

    http.post.assert_called_once_with(
        "/v2/enqueue",
        json={
            "routing_key": "routing-key",
            "event_action": "acknowledge",
            "dedup_key": "dedup-123",
            "payload": {"summary": "on it"},
        },
    )
