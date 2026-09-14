# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Default ADMISSION / COMPLETION mapping for observability delivery."""

from __future__ import annotations

from intergrax.contracts.event_delivery import (
    EventDeliveryObligation,
    EventDeliveryObligationPolicyPort,
    EventPriority,
)


class EnterpriseDefaultEventDeliveryObligationPolicy:
    """BEST_EFFORT/IMPORTANT → ADMISSION; CRITICAL → COMPLETION (before platform floor)."""

    def obligation_for(self, priority: EventPriority) -> EventDeliveryObligation:
        if priority is EventPriority.CRITICAL:
            return EventDeliveryObligation.COMPLETION
        return EventDeliveryObligation.ADMISSION
