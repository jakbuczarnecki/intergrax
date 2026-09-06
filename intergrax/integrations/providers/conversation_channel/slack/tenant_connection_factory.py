# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Slack tenant connection factory for Vendor Knowledge rehydration."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping

from intergrax.integrations.contracts.base import (
    IntegrationCategory,
    IntegrationConfigurationError,
)
from intergrax.runtime.vendor_knowledge.tenant_connection_factory_contract import (
    EagerTenantConnectionIntegrationFactoryMixin,
)
from intergrax.integrations.providers.conversation_channel.slack.config import (
    SlackConversationChannelIntegrationConfig,
)
from intergrax.integrations.providers.conversation_channel.slack.integration import (
    SLACK_CONVERSATION_CHANNEL_PROVIDER_ID,
    SlackConversationChannelIntegration,
)
from intergrax.runtime.vendor_knowledge.models import JsonValue

_ALLOWED_SECRET_FREE_CONFIG_KEYS = frozenset({"api_timeout_seconds"})
_DEFAULT_API_TIMEOUT_SECONDS = 30.0

SlackRuntimeBuilder = Callable[
    [SlackConversationChannelIntegrationConfig],
    SlackConversationChannelIntegration,
]


def _require_nonblank(value: str, *, field_name: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field_name} must be a non-empty string")
    return cleaned


def _parse_credentials(credential: str) -> tuple[str, str]:
    try:
        parsed = json.loads(credential)
    except json.JSONDecodeError:
        raise ValueError("Slack credential payload is not valid JSON") from None
    if not isinstance(parsed, dict):
        raise ValueError("Slack credential payload must be a JSON object")

    app_token = parsed.get("app_token")
    bot_token = parsed.get("bot_token")
    if not isinstance(app_token, str) or not app_token.strip():
        raise ValueError("Slack credential payload is missing app_token")
    if not isinstance(bot_token, str) or not bot_token.strip():
        raise ValueError("Slack credential payload is missing bot_token")
    return app_token, bot_token


def _resolve_timeout(secret_free_config: Mapping[str, JsonValue]) -> float:
    unexpected = set(secret_free_config) - _ALLOWED_SECRET_FREE_CONFIG_KEYS
    if unexpected:
        raise ValueError("Slack secret-free configuration contains unsupported fields")
    raw_timeout = secret_free_config.get(
        "api_timeout_seconds",
        _DEFAULT_API_TIMEOUT_SECONDS,
    )
    if isinstance(raw_timeout, bool) or not isinstance(raw_timeout, (int, float)):
        raise ValueError("Slack api_timeout_seconds must be a number")
    return float(raw_timeout)


class SlackTenantConnectionIntegrationFactory(
    EagerTenantConnectionIntegrationFactoryMixin,
):
    """Build Slack runtime integrations from secret-ref-backed credentials."""

    def __init__(self, runtime_builder: SlackRuntimeBuilder | None = None) -> None:
        self._runtime_builder = runtime_builder or SlackConversationChannelIntegration.from_config

    def create_integration(
        self,
        *,
        tenant_id: str,
        connection_ref: str,
        provider_id: str,
        integration_kind: IntegrationCategory,
        credential_ref: str,
        credential: str,
        secret_free_config: Mapping[str, JsonValue],
    ) -> SlackConversationChannelIntegration:
        return self.create_integration_with_resolved_credential(
            tenant_id=tenant_id,
            connection_ref=connection_ref,
            provider_id=provider_id,
            integration_kind=integration_kind,
            credential_ref=credential_ref,
            resolved_credential=credential,
            secret_free_config=secret_free_config,
        )

    def create_integration_with_resolved_credential(
        self,
        *,
        tenant_id: str,
        connection_ref: str,
        provider_id: str,
        integration_kind: IntegrationCategory,
        credential_ref: str,
        resolved_credential: str,
        secret_free_config: Mapping[str, JsonValue],
    ) -> SlackConversationChannelIntegration:
        _require_nonblank(tenant_id, field_name="tenant_id")
        _require_nonblank(connection_ref, field_name="connection_ref")
        _require_nonblank(
            credential_ref,
            field_name="credential_ref",
        )
        if provider_id != SLACK_CONVERSATION_CHANNEL_PROVIDER_ID:
            raise ValueError("provider_id does not match slack")
        if integration_kind is not IntegrationCategory.CONVERSATION_CHANNEL:
            raise ValueError("integration_kind does not match conversation_channel")
        if not isinstance(resolved_credential, str) or not resolved_credential.strip():
            raise ValueError("credential must be a nonblank string")

        app_token, bot_token = _parse_credentials(resolved_credential)
        timeout = _resolve_timeout(secret_free_config)
        config = SlackConversationChannelIntegrationConfig(
            enabled=True,
            app_token=app_token,
            bot_token=bot_token,
            api_timeout_seconds=timeout,
        )
        config.validate_for_runtime()
        try:
            return self._runtime_builder(config)
        except IntegrationConfigurationError:
            raise
        except Exception as exc:  # noqa: BLE001 — normalize provider construction errors
            raise IntegrationConfigurationError(
                "Slack conversation runtime construction failed (credentials redacted)",
            ) from exc


SlackConversationChannelTenantConnectionIntegrationFactory = (
    SlackTenantConnectionIntegrationFactory
)


__all__ = [
    "SlackConversationChannelTenantConnectionIntegrationFactory",
    "SlackTenantConnectionIntegrationFactory",
]
