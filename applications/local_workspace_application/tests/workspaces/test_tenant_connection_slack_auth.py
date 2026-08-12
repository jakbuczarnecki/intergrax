# © Artur Czarnecki. All rights reserved.

"""Slack manual credential binding adapter tests (PRODUCT-5B)."""

from __future__ import annotations

import pytest

from intergrax.integrations.providers.conversation_channel.slack.tenant_connection_auth import (
    SlackTenantConnectionAuthProvider,
)
from intergrax.runtime.vendor_knowledge.tenant_connection_auth import TenantConnectionAuthQualification

pytestmark = pytest.mark.unit


def test_slack_manual_binding_validates_shape() -> None:
    provider = SlackTenantConnectionAuthProvider()
    assert provider.qualification is TenantConnectionAuthQualification.NOT_QUALIFIED
    result = provider.bind_manual_credentials(
        tenant_id="tenant-a",
        credential_payload={
            "app_token": "xapp-1",
            "bot_token": "xoxb-1",
        },
    )
    assert "xapp-1" in result.credential_bundle_json
    assert result.connected_principal_ref is None


def test_slack_rejects_invalid_tokens() -> None:
    provider = SlackTenantConnectionAuthProvider()
    with pytest.raises(ValueError, match="credential_binding_invalid"):
        provider.bind_manual_credentials(
            tenant_id="tenant-a",
            credential_payload={"app_token": "bad", "bot_token": "xoxb-1"},
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
