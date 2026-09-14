# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Default late-failure observer — diagnostic only, never fails execution."""

from __future__ import annotations

import logging

from intergrax.contracts.event_delivery import (
    EventDeliveryLateFailure,
    EventDeliveryPostAdmissionFailureObserverPort,
)

logger = logging.getLogger(__name__)


class LoggingEventDeliveryPostAdmissionFailureObserver:
    def on_late_failure(self, failure: EventDeliveryLateFailure) -> None:
        logger.warning(
            "observability delivery late failure event_id=%s priority=%s disposition=%s stage=%s boundary=%s",
            failure.deliverable.event_id,
            failure.priority.value,
            failure.disposition.value,
            failure.stage.value,
            failure.boundary_kind.value if failure.boundary_kind is not None else "",
        )
