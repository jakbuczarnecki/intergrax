# © Artur Czarnecki. All rights reserved.

"""M365 OAuth adapter unit tests (PRODUCT-5B)."""

from __future__ import annotations

import pytest

from intergrax.integrations.providers.collaboration_suite.ms365_graph.tenant_connection_auth import (
    Ms365GraphOAuthConfig,
    Ms365GraphTenantConnectionAuthProvider,
)

pytestmark = pytest.mark.unit


def test_m365_begin_oauth_url() -> None:
    provider = Ms365GraphTenantConnectionAuthProvider(
        Ms365GraphOAuthConfig(
            tenant_id="tenant-guid",
            client_id="client-id",
            client_secret="secret",
        ),
    )
    begin = provider.begin_authorization(
        tenant_id="tenant-a",
        redirect_uri="https://app.example/callback",
        reconnect_connection_ref=None,
    )
    assert begin.authorization_url is not None
    assert "login.microsoftonline.com" in begin.authorization_url
    assert begin.code_verifier is not None


def test_m365_secret_free_config_includes_client_id() -> None:
    provider = Ms365GraphTenantConnectionAuthProvider(
        Ms365GraphOAuthConfig(
            tenant_id="tenant-guid",
            client_id="client-id",
            client_secret="secret",
        ),
    )
    config = provider.build_secret_free_config(tenant_id="tenant-a", reconnect_connection=None)
    assert config["client_id"] == "client-id"
    assert "client_secret" not in config
