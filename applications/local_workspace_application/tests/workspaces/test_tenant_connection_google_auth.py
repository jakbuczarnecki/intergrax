# © Artur Czarnecki. All rights reserved.

"""Google OAuth adapter unit tests (PRODUCT-5B)."""

from __future__ import annotations

import pytest

from intergrax.integrations.providers.collaboration_suite.google_workspace.tenant_connection_auth import (
    GoogleWorkspaceOAuthConfig,
    GoogleWorkspaceTenantConnectionAuthProvider,
)
from intergrax.runtime.vendor_knowledge.tenant_connection_auth import generate_pkce_pair

pytestmark = pytest.mark.unit


def test_google_begin_includes_pkce_challenge() -> None:
    provider = GoogleWorkspaceTenantConnectionAuthProvider(
        GoogleWorkspaceOAuthConfig(client_id="client-id", client_secret="secret"),
    )
    begin = provider.begin_authorization(
        tenant_id="tenant-a",
        redirect_uri="https://app.example/callback",
        reconnect_connection_ref=None,
    )
    assert begin.code_verifier is not None
    assert begin.authorization_url is not None
    assert "code_challenge=" in begin.authorization_url
    verifier, challenge = generate_pkce_pair()
    _ = verifier, challenge
    assert "code_challenge_method=S256" in begin.authorization_url or "S256" in begin.authorization_url


def test_google_misconfigured_without_client() -> None:
    provider = GoogleWorkspaceTenantConnectionAuthProvider(None)
    with pytest.raises(ValueError, match="connection_provider_misconfigured"):
        provider.begin_authorization(
            tenant_id="tenant-a",
            redirect_uri="https://app.example/callback",
            reconnect_connection_ref=None,
        )
