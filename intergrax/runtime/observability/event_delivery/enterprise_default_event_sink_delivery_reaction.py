# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Default delivery reaction policy (enterprise-safe baseline)."""

from __future__ import annotations

from intergrax.contracts.event_delivery import (
    DeliverableEvent,
    EventDeliveryBoundaryError,
    EventDeliveryDisposition,
    EventDeliveryReaction,
    EventDeliveryResult,
    EventPriority,
    EventSinkDeliveryReactionPort,
)


class EnterpriseDefaultEventSinkDeliveryReaction:
    """Preserves pre-P1B-R3 behavior for non-critical events; critical uses bus invariants."""

    def react_to_result(
        self,
        *,
        priority: EventPriority,
        result: EventDeliveryResult,
        deliverable: DeliverableEvent,
    ) -> EventDeliveryReaction:
        _ = deliverable
        if priority is EventPriority.CRITICAL and result.disposition in (
            EventDeliveryDisposition.DROPPED,
            EventDeliveryDisposition.REJECTED,
        ):
            return EventDeliveryReaction.FAIL_EXECUTION
        return EventDeliveryReaction.CONTINUE

    def react_to_boundary_error(
        self,
        *,
        priority: EventPriority,
        error: EventDeliveryBoundaryError,
        deliverable: DeliverableEvent,
    ) -> EventDeliveryReaction:
        _ = error
        _ = deliverable
        if priority is EventPriority.CRITICAL:
            return EventDeliveryReaction.FAIL_EXECUTION
        return EventDeliveryReaction.CONTINUE
