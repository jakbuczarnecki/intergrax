# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Default delivery-buffer admission limits (reserved CRITICAL capacity)."""

from __future__ import annotations

from intergrax.contracts.event_delivery import EventDeliveryPolicy


def enterprise_default_critical_reserved_slots(total_capacity: int) -> int:
    """Derive a bounded CRITICAL reserve from total buffer capacity (enterprise wiring)."""
    if total_capacity <= 1:
        return 1
    reserve = max(1, total_capacity // 16)
    return min(reserve, total_capacity)


class EnterpriseDefaultEventDeliveryAdmissionPolicy:
    """Lower priorities share ``max_capacity - critical_reserved_capacity`` queue slots."""

    def max_non_critical_buffered_events(self, policy: EventDeliveryPolicy) -> int:
        return policy.max_capacity - policy.critical_reserved_capacity
