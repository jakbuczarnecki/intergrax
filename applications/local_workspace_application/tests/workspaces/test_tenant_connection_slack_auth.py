# © Artur Czarnecki. All rights reserved.

"""Slack manual credential binding adapter tests (PRODUCT-5B/5D)."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from intergrax.integrations.providers.conversation_channel.slack.tenant_connection_auth import (
    SlackTenantConnectionAuthProvider,
)
from intergrax.runtime.vendor_knowledge.tenant_connection_auth import TenantConnectionAuthQualification

pytestmark = pytest.mark.unit


def test_slack_manual_binding_validates_shape_and_identity() -> None:
    provider = SlackTenantConnectionAuthProvider()
    assert provider.qualification is TenantConnectionAuthQualification.QUALIFIED
    with patch(
        "intergrax.integrations.providers.conversation_channel.slack.tenant_connection_auth._http_post_auth_test",
        return_value={"ok": True, "team_id": "T01234567"},
    ):
        result = provider.bind_manual_credentials(
            tenant_id="tenant-a",
            credential_payload={
                "app_token": "xapp-1",
                "bot_token": "xoxb-1",
            },
        )
    assert "xapp-1" in result.credential_bundle_json
    assert result.connected_principal_ref == "slack_team:T01234567"


def test_slack_rejects_invalid_tokens() -> None:
    provider = SlackTenantConnectionAuthProvider()
    with pytest.raises(ValueError, match="credential_binding_invalid"):
        provider.bind_manual_credentials(
            tenant_id="tenant-a",
            credential_payload={"app_token": "bad", "bot_token": "xoxb-1"},
        )


def test_slack_rejects_invalid_remote_credentials() -> None:
    provider = SlackTenantConnectionAuthProvider()
    with patch(
        "intergrax.integrations.providers.conversation_channel.slack.tenant_connection_auth._http_post_auth_test",
        return_value={"ok": False, "error": "invalid_auth"},
    ):
        with pytest.raises(ValueError, match="credential_binding_invalid"):
            provider.bind_manual_credentials(
                tenant_id="tenant-a",
                credential_payload={"app_token": "xapp-1", "bot_token": "xoxb-1"},
            )


def test_slack_begin_manual_instructions() -> None:
    provider = SlackTenantConnectionAuthProvider()
    begin = provider.begin_authorization(
        tenant_id="tenant-a",
        redirect_uri="",
        reconnect_connection_ref=None,
    )
    assert begin.required_user_action == "present_manual_instructions"
    assert begin.authorization_url is None
