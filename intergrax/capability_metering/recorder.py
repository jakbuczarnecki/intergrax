# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Validated capability usage recorder (CAPABILITY-CATALOG-1 Stage 13)."""

from __future__ import annotations

from intergrax.capability_metering.attribution import CapabilityUsageAttribution
from intergrax.capability_metering.consumer import CapabilityUsageConsumer
from intergrax.contracts.capability_metering import (
    CapabilityUsageEvent,
    CapabilityUsageKind,
    CapabilityUsageOutcome,
    build_capability_usage_event,
)
from intergrax.contracts.execution_identity import AttemptId, RunId, TaskId


class CapabilityUsageRecorder:
    """Builds validated usage evidence and forwards to an injected consumer."""

    def __init__(self, consumer: CapabilityUsageConsumer | None = None) -> None:
        self._consumer = consumer

    def record(
        self,
        *,
        tenant_id: str,
        attribution: CapabilityUsageAttribution,
        usage_kind: CapabilityUsageKind,
        outcome: CapabilityUsageOutcome,
        quantity: int = 1,
        task_id: TaskId | None = None,
        run_id: RunId | None = None,
        attempt_id: AttemptId | None = None,
    ) -> CapabilityUsageEvent:
        event = build_capability_usage_event(
            tenant_id=tenant_id,
            identity=attribution.identity,
            provenance=attribution.provenance,
            usage_kind=usage_kind,
            outcome=outcome,
            quantity=quantity,
            task_id=task_id,
            run_id=run_id,
            attempt_id=attempt_id,
        )
        if self._consumer is not None:
            self._consumer.consume(event)
        return event
