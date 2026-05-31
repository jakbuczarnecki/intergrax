# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Resilient notification delivery wiring (Appendix B.13)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from intergrax.integrations.providers.relational_store.sqlite.opens import open_delivery_ledger_at
from intergrax.runtime.notifications.deliveries.delivery_ledger import InMemoryDeliveryLedger
from intergrax.runtime.notifications.deliveries.delivery_ledger_protocol import DeliveryLedger
from intergrax.runtime.notifications.deliveries.retry_delivery import RetryingNotificationDelivery
from intergrax.runtime.notifications.delivery_contract import NotificationDelivery


def open_delivery_ledger(
    *,
    db_path: Path | None = None,
    in_memory: bool = False,
) -> DeliveryLedger:
    if in_memory or db_path is None:
        return InMemoryDeliveryLedger()
    return open_delivery_ledger_at(db_path)


def create_resilient_delivery(
    inner: NotificationDelivery,
    *,
    ledger: Optional[DeliveryLedger] = None,
    max_attempts: int = 3,
    channel: str = "webhook",
) -> NotificationDelivery:
    if ledger is None:
        return inner
    return RetryingNotificationDelivery(
        inner,
        max_attempts=max_attempts,
        ledger=ledger,  # type: ignore[arg-type]
        channel=channel,
    )
