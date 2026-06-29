# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""
Complete Teams integration bundle — the single composition root for Microsoft Teams.

Outbound (HITL / escalation) and inbound (Bot Framework activities) wiring MUST use
this module or ``profile.resolve()`` with ``"teams"``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationConfigurationError
from intergrax.integrations.providers.notification_channel.teams.config import TeamsIntegrationConfig
from intergrax.integrations.providers.notification_channel.teams.opens import (
    open_teams_interaction_surface,
    open_teams_notification_channel,
)
from intergrax.runtime.interactions.adapter_contract import InteractionAdapter
from intergrax.runtime.notifications.adapter_contract import NotificationAdapter
from intergrax.runtime.notifications.delivery_contract import NotificationDelivery


@dataclass(frozen=True)
class TeamsIntegrationBundle:
    """Teams notification + interaction adapters sharing one config."""

    config: TeamsIntegrationConfig
    notification_channel: NotificationAdapter
    interaction_surface: InteractionAdapter


def resolve_teams_config(**overrides: object) -> TeamsIntegrationConfig:
    return TeamsIntegrationConfig.from_env(**overrides)


def create_teams_integration(
    *,
    webhook_url: Optional[str] = None,
    security_token: Optional[str] = None,
    notification_adapter: Optional[NotificationAdapter] = None,
    interaction_adapter: Optional[InteractionAdapter] = None,
    delivery: Optional[NotificationDelivery] = None,
    **config_overrides: object,
) -> TeamsIntegrationBundle:
    """Single entry point for Teams — outbound webhook + inbound activities."""
    overrides: dict[str, object] = dict(config_overrides)
    if webhook_url is not None:
        overrides["webhook_url"] = webhook_url
    if security_token is not None:
        overrides["security_token"] = security_token

    config = resolve_teams_config(**overrides)
    notification = open_teams_notification_channel(
        config,
        implementation=notification_adapter,
        delivery=delivery,
    )
    interaction = open_teams_interaction_surface(
        config,
        implementation=interaction_adapter,
    )

    return TeamsIntegrationBundle(
        config=config,
        notification_channel=notification,
        interaction_surface=interaction,
    )


def create_teams_notification_channel(
    *,
    webhook_url: Optional[str] = None,
    notification_adapter: Optional[NotificationAdapter] = None,
    delivery: Optional[NotificationDelivery] = None,
    **config_overrides: object,
) -> NotificationAdapter:
    """Direct factory for outbound Teams notifications."""
    return create_teams_integration(
        webhook_url=webhook_url,
        notification_adapter=notification_adapter,
        delivery=delivery,
        **config_overrides,
    ).notification_channel


def create_teams_interaction_surface(
    *,
    interaction_adapter: Optional[InteractionAdapter] = None,
    **config_overrides: object,
) -> InteractionAdapter:
    """Direct factory for inbound Teams Bot Framework payloads."""
    return create_teams_integration(
        interaction_adapter=interaction_adapter,
        **config_overrides,
    ).interaction_surface


def create_teams_catalog_factory(
    *,
    integration_category: IntegrationCategory,
    **config_overrides: object,
) -> Any:
    """Catalog factory — dispatches by resolved category (dual-role slug)."""
    bundle = create_teams_integration(**config_overrides)
    if integration_category == IntegrationCategory.NOTIFICATION_CHANNEL:
        return bundle.notification_channel
    if integration_category == IntegrationCategory.INTERACTION_SURFACE:
        return bundle.interaction_surface
    raise IntegrationConfigurationError(
        f"Teams integration does not support category '{integration_category.value}'."
    )


def create_teams_signature_verifier(
    *,
    security_token: Optional[str] = None,
    enabled: Optional[bool] = None,
) -> object:
    """Inbound HTTP verification — wraps ``TeamsSignatureVerifier``."""
    from intergrax.runtime.interactions.verification.teams_signature import (
        TeamsSignatureVerifier,
        resolve_teams_security_token,
    )

    return TeamsSignatureVerifier(
        security_token=resolve_teams_security_token(security_token),
        enabled=enabled,
    )


from intergrax.integrations.providers.notification_channel.teams.integration import (
    TEAMS_NOTIFICATION_CHANNEL_PROVIDER_ID,
    TeamsNotificationChannelClient,
    TeamsNotificationChannelIntegration,
    TeamsNotificationChannelIntegrationConfig,
)


def create_teams_notification_channel_integration(
    *,
    client: TeamsNotificationChannelClient | None = None,
    enabled: bool = False,
) -> TeamsNotificationChannelIntegration:
    """
    Build a contract-based Teams notification channel integration.

    Compatibility shim — constructs Integration via from_store (create_teams_integration) is unchanged.
    Client must be injected explicitly when enabled=True; disabled by default.
    """
    if enabled and client is None:
        raise IntegrationConfigurationError(
            "Teams notification channel integration requires an injected client when enabled=True",
        )
    if client is not None:
        return TeamsNotificationChannelIntegration.from_client(client, enabled=enabled)
    return TeamsNotificationChannelIntegration.for_provider(
        provider_id=TEAMS_NOTIFICATION_CHANNEL_PROVIDER_ID,
        display_name="Teams",
        config=TeamsNotificationChannelIntegrationConfig(enabled=enabled),
    )
