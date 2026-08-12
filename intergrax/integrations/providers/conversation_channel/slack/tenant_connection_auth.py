# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Slack manual credential binding auth adapter (PRODUCT-5B)."""

from __future__ import annotations

import json
from collections.abc import Mapping

from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.providers.conversation_channel.slack.integration import (
    SLACK_CONVERSATION_CHANNEL_PROVIDER_ID,
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
)


class SlackTenantConnectionAuthProvider:
    """Manual app/bot token binding for Slack knowledge source connections."""

    provider_id = SLACK_CONVERSATION_CHANNEL_PROVIDER_ID
    integration_kind = IntegrationCategory.CONVERSATION_CHANNEL
    auth_mode = TenantConnectionAuthMode.MANUAL_CREDENTIAL_BINDING
    qualification = TenantConnectionAuthQualification.NOT_QUALIFIED

    def describe(self) -> TenantConnectionAuthProviderDescriptor:
        return TenantConnectionAuthProviderDescriptor(
            provider_id=self.provider_id,
            integration_kind=self.integration_kind,
            auth_mode=self.auth_mode,
            safe_display_name="Slack",
            supported_scopes_summary="Workspace channels and messages (manual tokens)",
            qualification=self.qualification,
        )

    def begin_authorization(
        self,
        *,
        tenant_id: str,
        redirect_uri: str,
        reconnect_connection_ref: str | None,
    ) -> TenantConnectionAuthBeginResult:
        _ = tenant_id, redirect_uri, reconnect_connection_ref
        return TenantConnectionAuthBeginResult(
            authorization_url=None,
            code_verifier=None,
            correlation_state=generate_correlation_state(),
            required_user_action="present_manual_instructions",
            manual_instructions=(
                "Provide Slack app token (xapp-) and bot token (xoxb-) as JSON: "
                '{"app_token":"...","bot_token":"..."}'
            ),
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
        _ = tenant_id, redirect_uri, authorization_code, code_verifier, correlation_state
        raise ValueError("Slack knowledge source does not support OAuth code exchange")

    def bind_manual_credentials(
        self,
        *,
        tenant_id: str,
        credential_payload: Mapping[str, JsonValue],
    ) -> TenantConnectionAuthManualBindResult:
        _ = tenant_id
        app_token = credential_payload.get("app_token")
        bot_token = credential_payload.get("bot_token")
        if not isinstance(app_token, str) or not app_token.strip().startswith("xapp-"):
            raise ValueError("credential_binding_invalid")
        if not isinstance(bot_token, str) or not bot_token.strip().startswith("xoxb-"):
            raise ValueError("credential_binding_invalid")
        bundle = json.dumps(
            {"app_token": app_token.strip(), "bot_token": bot_token.strip()},
            sort_keys=True,
        )
        return TenantConnectionAuthManualBindResult(
            credential_bundle_json=bundle,
            connected_principal_ref=None,
        )

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
        _ = tenant_id, credential_bundle_json


__all__ = ["SlackTenantConnectionAuthProvider"]
