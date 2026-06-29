# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""
Complete Slack integration bundle — the single composition root for Slack in Intergrax.

Outbound (HITL / escalation) and inbound (slash commands) wiring MUST use this module
or ``profile.resolve()`` with ``"slack"``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationConfigurationError
from intergrax.integrations.providers.notification_channel.slack.config import SlackIntegrationConfig
from intergrax.integrations.providers.notification_channel.slack.opens import (
    open_slack_interaction_surface,
    open_slack_notification_channel,
)
from intergrax.runtime.interactions.adapter_contract import InteractionAdapter
from intergrax.runtime.notifications.adapter_contract import NotificationAdapter
from intergrax.runtime.notifications.delivery_contract import NotificationDelivery


@dataclass(frozen=True)
class SlackIntegrationBundle:
    """Slack notification + interaction adapters sharing one config."""

    config: SlackIntegrationConfig
    notification_channel: NotificationAdapter
    interaction_surface: InteractionAdapter


def resolve_slack_config(**overrides: object) -> SlackIntegrationConfig:
    return SlackIntegrationConfig.from_env(**overrides)


def create_slack_integration(
    *,
    webhook_url: Optional[str] = None,
    signing_secret: Optional[str] = None,
    notification_adapter: Optional[NotificationAdapter] = None,
    interaction_adapter: Optional[InteractionAdapter] = None,
    delivery: Optional[NotificationDelivery] = None,
    **config_overrides: object,
) -> SlackIntegrationBundle:
    """Single entry point for Slack — outbound webhook + inbound slash commands."""
    overrides: dict[str, object] = dict(config_overrides)
    if webhook_url is not None:
        overrides["webhook_url"] = webhook_url
    if signing_secret is not None:
        overrides["signing_secret"] = signing_secret

    config = resolve_slack_config(**overrides)
    notification = open_slack_notification_channel(
        config,
        implementation=notification_adapter,
        delivery=delivery,
    )
    interaction = open_slack_interaction_surface(
        config,
        implementation=interaction_adapter,
    )

    return SlackIntegrationBundle(
        config=config,
        notification_channel=notification,
        interaction_surface=interaction,
    )


def create_slack_notification_channel(
    *,
    webhook_url: Optional[str] = None,
    notification_adapter: Optional[NotificationAdapter] = None,
    delivery: Optional[NotificationDelivery] = None,
    **config_overrides: object,
) -> NotificationAdapter:
    """Direct factory for outbound Slack notifications."""
    return create_slack_integration(
        webhook_url=webhook_url,
        notification_adapter=notification_adapter,
        delivery=delivery,
        **config_overrides,
    ).notification_channel


def create_slack_interaction_surface(
    *,
    interaction_adapter: Optional[InteractionAdapter] = None,
    **config_overrides: object,
) -> InteractionAdapter:
    """Direct factory for inbound Slack slash-command payloads."""
    return create_slack_integration(
        interaction_adapter=interaction_adapter,
        **config_overrides,
    ).interaction_surface


def create_slack_catalog_factory(
    *,
    integration_category: IntegrationCategory,
    **config_overrides: object,
) -> Any:
    """Catalog factory — dispatches by resolved category (dual-role slug)."""
    bundle = create_slack_integration(**config_overrides)
    if integration_category == IntegrationCategory.NOTIFICATION_CHANNEL:
        return bundle.notification_channel
    if integration_category == IntegrationCategory.INTERACTION_SURFACE:
        return bundle.interaction_surface
    raise IntegrationConfigurationError(
        f"Slack integration does not support category '{integration_category.value}'."
    )


def create_slack_signature_verifier(
    *,
    signing_secret: Optional[str] = None,
    enabled: Optional[bool] = None,
) -> object:
    """Inbound HTTP verification — wraps ``SlackSignatureVerifier``."""
    from intergrax.runtime.interactions.verification.slack_signature import (
        SlackSignatureVerifier,
        resolve_slack_signing_secret,
    )

    return SlackSignatureVerifier(
        signing_secret=resolve_slack_signing_secret(signing_secret),
        enabled=enabled,
    )


from intergrax.integrations.providers.notification_channel.slack.integration import (
    SLACK_NOTIFICATION_CHANNEL_PROVIDER_ID,
    SlackNotificationChannelClient,
    SlackNotificationChannelIntegration,
    SlackNotificationChannelIntegrationConfig,
)


def create_slack_notification_channel_integration(
    *,
    client: SlackNotificationChannelClient | None = None,
    enabled: bool = False,
) -> SlackNotificationChannelIntegration:
    """
    Build a contract-based Slack notification channel integration.

    Compatibility shim — constructs Integration via from_store (create_slack_integration) is unchanged.
    Client must be injected explicitly when enabled=True; disabled by default.
    """
    if enabled and client is None:
        raise IntegrationConfigurationError(
            "Slack notification channel integration requires an injected client when enabled=True",
        )
    if client is not None:
        return SlackNotificationChannelIntegration.from_client(client, enabled=enabled)
    return SlackNotificationChannelIntegration.for_provider(
        provider_id=SLACK_NOTIFICATION_CHANNEL_PROVIDER_ID,
        display_name="Slack",
        config=SlackNotificationChannelIntegrationConfig(enabled=enabled),
    )
