# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Microsoft 365 delegated OAuth auth adapter (PRODUCT-5B)."""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.providers.collaboration_suite.ms365_graph.integration import (
    MS365_GRAPH_COLLABORATION_SUITE_PROVIDER_ID,
)
from intergrax.integrations.providers.collaboration_suite.ms365_graph.config import (
    DEFAULT_GRAPH_BASE_URL,
    DEFAULT_TIMEOUT_SECONDS,
)
from intergrax.runtime.vendor_knowledge.models import JsonValue
from intergrax.runtime.vendor_knowledge.tenant_connection_auth import (
    TenantConnectionAuthBeginResult,
    TenantConnectionAuthExchangeResult,
    TenantConnectionAuthManualBindResult,
    TenantConnectionAuthMode,
    TenantConnectionAuthProviderDescriptor,
    TenantConnectionAuthQualification,
    generate_correlation_state,
    generate_pkce_pair,
)

_DEFAULT_SCOPES = (
    "openid",
    "profile",
    "offline_access",
    "User.Read",
    "Mail.Read",
    "Calendars.Read",
    "Files.Read.All",
)


@dataclass(frozen=True, slots=True)
class Ms365GraphOAuthConfig:
    tenant_id: str
    client_id: str
    client_secret: str
    graph_base_url: str = DEFAULT_GRAPH_BASE_URL
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    scopes: tuple[str, ...] = _DEFAULT_SCOPES

    @property
    def authorize_url(self) -> str:
        return (
            f"https://login.microsoftonline.com/{self.tenant_id.strip()}/oauth2/v2.0/authorize"
        )

    @property
    def token_url(self) -> str:
        return (
            f"https://login.microsoftonline.com/{self.tenant_id.strip()}/oauth2/v2.0/token"
        )


def _http_post_json(
    url: str,
    payload: Mapping[str, str],
    timeout: float = 30.0,
) -> dict[str, object]:
    body = urllib.parse.urlencode(dict(payload)).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={"Accept": "application/json", "Content-Type": "application/x-www-form-urlencoded"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read().decode("utf-8")
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("provider response is invalid")
    return parsed


def _http_get_json(url: str, access_token: str, timeout: float = 30.0) -> dict[str, object]:
    request = urllib.request.Request(
        url,
        method="GET",
        headers={"Accept": "application/json", "Authorization": f"Bearer {access_token}"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read().decode("utf-8")
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("provider response is invalid")
    return parsed


class Ms365GraphTenantConnectionAuthProvider:
    """Delegated OAuth for Microsoft 365 Graph tenant connections."""

    provider_id = MS365_GRAPH_COLLABORATION_SUITE_PROVIDER_ID
    integration_kind = IntegrationCategory.COLLABORATION_SUITE
    auth_mode = TenantConnectionAuthMode.OAUTH_DELEGATED
    qualification = TenantConnectionAuthQualification.QUALIFIED

    def __init__(self, oauth_config: Ms365GraphOAuthConfig | None) -> None:
        self._oauth_config = oauth_config

    def describe(self) -> TenantConnectionAuthProviderDescriptor:
        return TenantConnectionAuthProviderDescriptor(
            provider_id=self.provider_id,
            integration_kind=self.integration_kind,
            auth_mode=self.auth_mode,
            safe_display_name="Microsoft 365",
            supported_scopes_summary="Mail, Calendar, OneDrive (delegated read)",
            qualification=self.qualification,
        )

    def _require_config(self) -> Ms365GraphOAuthConfig:
        if self._oauth_config is None:
            raise ValueError("connection_provider_misconfigured")
        config = self._oauth_config
        if (
            not config.tenant_id.strip()
            or not config.client_id.strip()
            or not config.client_secret.strip()
        ):
            raise ValueError("connection_provider_misconfigured")
        return config

    def begin_authorization(
        self,
        *,
        tenant_id: str,
        redirect_uri: str,
        reconnect_connection_ref: str | None,
    ) -> TenantConnectionAuthBeginResult:
        _ = tenant_id, reconnect_connection_ref
        config = self._require_config()
        verifier, challenge = generate_pkce_pair()
        correlation_state = generate_correlation_state()
        params = {
            "client_id": config.client_id,
            "redirect_uri": redirect_uri,
            "response_type": "code",
            "scope": " ".join(config.scopes),
            "state": correlation_state,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
            "response_mode": "query",
        }
        authorization_url = f"{config.authorize_url}?{urllib.parse.urlencode(params)}"
        return TenantConnectionAuthBeginResult(
            authorization_url=authorization_url,
            code_verifier=verifier,
            correlation_state=correlation_state,
            required_user_action="redirect",
        )

    def exchange_authorization_code(
        self,
        *,
        tenant_id: str,
        redirect_uri: str,
        authorization_code: str,
        code_verifier: str,
        correlation_state: str,
    ) -> TenantConnectionAuthExchangeResult:
        _ = tenant_id, correlation_state
        config = self._require_config()
        token_payload = _http_post_json(
            config.token_url,
            {
                "client_id": config.client_id,
                "client_secret": config.client_secret,
                "code": authorization_code.strip(),
                "code_verifier": code_verifier,
                "grant_type": "authorization_code",
                "redirect_uri": redirect_uri,
            },
            timeout=config.timeout_seconds,
        )
        access_token = token_payload.get("access_token")
        if not isinstance(access_token, str) or not access_token:
            raise ValueError("provider token response is invalid")
        refresh_token = token_payload.get("refresh_token")
        expires_in = token_payload.get("expires_in")
        expires_at: str | None = None
        if isinstance(expires_in, (int, float)) and not isinstance(expires_in, bool):
            expires_at = (
                datetime.now(UTC) + timedelta(seconds=float(expires_in))
            ).isoformat()
        bundle: dict[str, str] = {"access_token": access_token}
        if isinstance(refresh_token, str) and refresh_token:
            bundle["refresh_token"] = refresh_token
        if expires_at is not None:
            bundle["expires_at"] = expires_at
        token_type = token_payload.get("token_type")
        if isinstance(token_type, str) and token_type:
            bundle["token_type"] = token_type
        scope = token_payload.get("scope")
        if isinstance(scope, str) and scope:
            bundle["scope"] = scope

        graph_base = config.graph_base_url.rstrip("/")
        profile = _http_get_json(f"{graph_base}/me", access_token, timeout=config.timeout_seconds)
        oid = profile.get("id")
        principal: str | None = None
        if isinstance(oid, str) and oid.strip():
            principal = oid.strip()
        return TenantConnectionAuthExchangeResult(
            credential_bundle_json=json.dumps(bundle, sort_keys=True),
            connected_principal_ref=principal,
        )

    def bind_manual_credentials(
        self,
        *,
        tenant_id: str,
        credential_payload: Mapping[str, JsonValue],
    ) -> TenantConnectionAuthManualBindResult:
        _ = tenant_id, credential_payload
        raise ValueError("Microsoft 365 does not support manual credential binding")

    def build_secret_free_config(
        self,
        *,
        tenant_id: str,
        reconnect_connection: object | None,
    ) -> Mapping[str, JsonValue]:
        _ = tenant_id
        config = self._oauth_config
        if config is None:
            raise ValueError("connection_provider_misconfigured")
        result: dict[str, JsonValue] = {
            "client_id": config.client_id,
            "graph_base_url": config.graph_base_url,
            "timeout_seconds": config.timeout_seconds,
        }
        if reconnect_connection is not None:
            principal = getattr(reconnect_connection, "connected_principal_ref", None)
            if isinstance(principal, str) and principal.strip():
                result["default_user"] = principal.strip()
        return result

    def revoke_remote_credentials(
        self,
        *,
        tenant_id: str,
        credential_bundle_json: str,
    ) -> None:
        _ = tenant_id
        try:
            parsed = json.loads(credential_bundle_json)
        except json.JSONDecodeError:
            return
        if not isinstance(parsed, dict):
            return
        # Best-effort local-only for delegated tokens; Graph lacks universal revoke for user tokens.
        _ = parsed


__all__ = [
    "Ms365GraphOAuthConfig",
    "Ms365GraphTenantConnectionAuthProvider",
]
