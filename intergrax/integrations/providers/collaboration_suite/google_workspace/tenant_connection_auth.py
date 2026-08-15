# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Google Workspace delegated OAuth auth adapter (PRODUCT-5B)."""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.providers.collaboration_suite.google_workspace.integration import (
    GOOGLE_WORKSPACE_COLLABORATION_SUITE_PROVIDER_ID,
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

_GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
_GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
_GOOGLE_USERINFO_URL = "https://openidconnect.googleapis.com/v1/userinfo"
_DEFAULT_SCOPES = (
    "openid",
    "email",
    "profile",
    "https://www.googleapis.com/auth/drive.readonly",
    "https://www.googleapis.com/auth/calendar.readonly",
    "https://www.googleapis.com/auth/documents.readonly",
    "https://www.googleapis.com/auth/spreadsheets.readonly",
)


@dataclass(frozen=True, slots=True)
class GoogleWorkspaceOAuthConfig:
    client_id: str
    client_secret: str
    scopes: tuple[str, ...] = _DEFAULT_SCOPES


def _http_post_json(url: str, payload: Mapping[str, str], timeout: float = 30.0) -> dict[str, object]:
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


class GoogleWorkspaceTenantConnectionAuthProvider:
    """Delegated OAuth for Google Workspace tenant connections."""

    provider_id = GOOGLE_WORKSPACE_COLLABORATION_SUITE_PROVIDER_ID
    integration_kind = IntegrationCategory.COLLABORATION_SUITE
    auth_mode = TenantConnectionAuthMode.OAUTH_DELEGATED
    qualification = TenantConnectionAuthQualification.QUALIFIED

    def __init__(self, oauth_config: GoogleWorkspaceOAuthConfig | None) -> None:
        self._oauth_config = oauth_config

    def describe(self) -> TenantConnectionAuthProviderDescriptor:
        return TenantConnectionAuthProviderDescriptor(
            provider_id=self.provider_id,
            integration_kind=self.integration_kind,
            auth_mode=self.auth_mode,
            safe_display_name="Google Workspace",
            supported_scopes_summary="Drive, Calendar, Docs, Sheets (read)",
            qualification=self.qualification,
        )

    def _require_config(self) -> GoogleWorkspaceOAuthConfig:
        if self._oauth_config is None:
            raise ValueError("connection_provider_misconfigured")
        if not self._oauth_config.client_id.strip() or not self._oauth_config.client_secret.strip():
            raise ValueError("connection_provider_misconfigured")
        return self._oauth_config

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
            "access_type": "offline",
            "prompt": "consent",
        }
        authorization_url = f"{_GOOGLE_AUTH_URL}?{urllib.parse.urlencode(params)}"
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
            _GOOGLE_TOKEN_URL,
            {
                "client_id": config.client_id,
                "client_secret": config.client_secret,
                "code": authorization_code.strip(),
                "code_verifier": code_verifier,
                "grant_type": "authorization_code",
                "redirect_uri": redirect_uri,
            },
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

        userinfo = _http_get_json(_GOOGLE_USERINFO_URL, access_token)
        sub = userinfo.get("sub")
        principal: str | None = None
        if isinstance(sub, str) and sub.strip():
            principal = sub.strip()
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
        raise ValueError("Google Workspace does not support manual credential binding")

    def build_secret_free_config(
        self,
        *,
        tenant_id: str,
        reconnect_connection: object | None,
    ) -> Mapping[str, JsonValue]:
        _ = tenant_id, reconnect_connection
        return {}

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
        token = parsed.get("refresh_token") or parsed.get("access_token")
        if not isinstance(token, str) or not token.strip():
            return
        config = self._oauth_config
        if config is None:
            return
        try:
            _http_post_json(
                _GOOGLE_TOKEN_URL,
                {
                    "token": token.strip(),
                    "client_id": config.client_id,
                    "client_secret": config.client_secret,
                },
            )
        except Exception:
            return


__all__ = [
    "GoogleWorkspaceOAuthConfig",
    "GoogleWorkspaceTenantConnectionAuthProvider",
]
