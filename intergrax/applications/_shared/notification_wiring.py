# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Shared notification + delivery ledger wiring for Tier-3 lab-style hosts."""

from __future__ import annotations

from pathlib import Path

from intergrax.applications._shared.wiring_posture import wiring_fails_closed
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.integrations.contracts.base import (
    IntegrationCategory,
    IntegrationConfigurationError,
    LabSelectorOptIn,
)
from intergrax.integrations.registry.factory import resolve_from_profile
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.runtime.notifications.adapter_contract import NotificationAdapter
from intergrax.runtime.notifications.backend_contract import NotificationBackend, resolve_notification_settings
from intergrax.runtime.notifications.deliveries.delivery_ledger_protocol import DeliveryLedger
from intergrax.runtime.notifications.deliveries.http_webhook_delivery import HttpWebhookDelivery
from intergrax.runtime.notifications.delivery_wiring import create_resilient_delivery, open_delivery_ledger
from intergrax.runtime.notifications.factory import create_notification_adapter


def resolve_delivery_ledger_db_path(
    *,
    db_path: Path | None,
    checkpoints_db_path: Path | None,
) -> Path | None:
    anchor = checkpoints_db_path or db_path
    if anchor is None:
        return None
    return anchor.parent / "intergrax_delivery_ledger.db"


def open_host_delivery_ledger(
    *,
    db_path: Path | None,
    checkpoints_db_path: Path | None,
) -> DeliveryLedger | None:
    ledger_db = resolve_delivery_ledger_db_path(
        db_path=db_path,
        checkpoints_db_path=checkpoints_db_path,
    )
    if ledger_db is None:
        return None
    return open_delivery_ledger(db_path=ledger_db)


_CATALOG_NOTIFICATION_SLUGS: dict[str] = {
    "log": "log",
    "slack": "slack",
    "teams": "teams",
    "webhook": "webhook",
    "pagerduty": "pagerduty",
    "opsgenie": "opsgenie",
    "discord": "discord",
    "twilio": "twilio",
    "email_smtp": "email_smtp",
}


def create_notification_adapter_from_profile(
    profile: IntegrationProfile,
    *,
    delivery_ledger: DeliveryLedger | None = None,
    lab_opt_in: LabSelectorOptIn | None = None,
) -> NotificationAdapter:
    """Resolve notification adapter via Integration Library catalog (Phase M.9)."""
    slug = profile.slug_for_category(IntegrationCategory.NOTIFICATION_CHANNEL)
    if slug is not None:
        return resolve_from_profile(profile, IntegrationCategory.NOTIFICATION_CHANNEL)  # type: ignore[return-value]
    return create_resilient_notification_adapter(
        profile,
        delivery_ledger=delivery_ledger,
        lab_opt_in=lab_opt_in,
    )


def resolve_host_notification_adapter(
    env: ApplicationEnvironmentProfile,
    *,
    integration_profile: IntegrationProfile | None = None,
    delivery_ledger: DeliveryLedger | None = None,
) -> NotificationAdapter:
    """Resolve a composition-time notification adapter for Tier-3 host wiring."""
    profile = integration_profile or env.integration_profile or IntegrationProfile()
    slug = profile.slug_for_category(IntegrationCategory.NOTIFICATION_CHANNEL)
    if slug is not None:
        return create_notification_adapter_from_profile(profile, delivery_ledger=delivery_ledger)

    if wiring_fails_closed(env):
        raise IntegrationConfigurationError("notification_channel binding is required for STRICT or PRODUCT posture")

    lab_opt_in = LabSelectorOptIn(
        selector=NotificationBackend.LOG.value,
        reason="derived laboratory notification default for non-strict host wiring",
    )
    return create_notification_adapter(lab_opt_in=lab_opt_in)


def create_harness_notification_adapter(profile: IntegrationProfile) -> NotificationAdapter:
    """Resolve notification channel directly from harness ``IntegrationProfile``."""
    return resolve_from_profile(profile, IntegrationCategory.NOTIFICATION_CHANNEL)  # type: ignore[return-value]


def create_resilient_notification_adapter(
    profile: IntegrationProfile,
    *,
    delivery_ledger: DeliveryLedger | None,
    lab_opt_in: LabSelectorOptIn | None = None,
) -> NotificationAdapter:
    slug = profile.slug_for_category(IntegrationCategory.NOTIFICATION_CHANNEL)
    if slug is not None:
        return resolve_from_profile(profile, IntegrationCategory.NOTIFICATION_CHANNEL)  # type: ignore[return-value]

    settings = resolve_notification_settings(lab_opt_in=lab_opt_in)
    backend_name = settings.backend.value
    if backend_name in _CATALOG_NOTIFICATION_SLUGS and settings.backend != NotificationBackend.LOG:
        resilient_delivery = create_resilient_delivery(
            HttpWebhookDelivery(),
            ledger=delivery_ledger,
            channel=backend_name,
        )
        return create_notification_adapter(settings, delivery=resilient_delivery)
    return create_notification_adapter(settings)
