# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Shared notification + delivery ledger wiring for Tier-3 lab-style hosts."""

from __future__ import annotations

from pathlib import Path

from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.runtime.notifications.adapter_contract import NotificationAdapter
from intergrax.runtime.notifications.deliveries.delivery_ledger_protocol import DeliveryLedger
from intergrax.runtime.notifications.deliveries.http_webhook_delivery import HttpWebhookDelivery
from intergrax.runtime.notifications.delivery_wiring import create_resilient_delivery, open_delivery_ledger
from intergrax.runtime.notifications.factory import (
    NotificationBackend,
    create_notification_adapter,
    resolve_notification_settings,
)


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
) -> NotificationAdapter:
    """Resolve notification adapter via Integration Library catalog (Phase M.9)."""
    slug = profile.slug_for_category(IntegrationCategory.NOTIFICATION_CHANNEL)
    if slug is None:
        return create_resilient_notification_adapter(profile, delivery_ledger=delivery_ledger)
    settings = resolve_notification_settings()
    backend_name = settings.backend.value
    catalog_slug = _CATALOG_NOTIFICATION_SLUGS.get(backend_name)
    if catalog_slug is not None and slug == catalog_slug:
        channel = profile.resolve(IntegrationCategory.NOTIFICATION_CHANNEL)
        return channel  # type: ignore[return-value]
    return create_resilient_notification_adapter(profile, delivery_ledger=delivery_ledger)


def create_harness_notification_adapter(profile: IntegrationProfile) -> NotificationAdapter:
    """Resolve notification channel directly from harness ``IntegrationProfile``."""
    return profile.resolve(IntegrationCategory.NOTIFICATION_CHANNEL)  # type: ignore[return-value]


def create_resilient_notification_adapter(
    profile: IntegrationProfile,
    *,
    delivery_ledger: DeliveryLedger | None,
) -> NotificationAdapter:
    settings = resolve_notification_settings()
    backend = settings.backend.value
    if backend in _CATALOG_NOTIFICATION_SLUGS:
        slug = profile.slug_for_category(IntegrationCategory.NOTIFICATION_CHANNEL)
        if slug == _CATALOG_NOTIFICATION_SLUGS[backend]:
            return profile.resolve(IntegrationCategory.NOTIFICATION_CHANNEL)  # type: ignore[return-value]
    if settings.backend == NotificationBackend.LOG:
        return profile.resolve(IntegrationCategory.NOTIFICATION_CHANNEL)  # type: ignore[return-value]

    resilient_delivery = create_resilient_delivery(
        HttpWebhookDelivery(),
        ledger=delivery_ledger,
        channel=settings.backend.value,
    )
    return create_notification_adapter(settings, delivery=resilient_delivery)
