# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Billing meter integration contract (Phase M.6 P6)."""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

from pydantic import BaseModel, Field


class MeterEvent(BaseModel):
    """Usage metering event row."""

    event_id: str
    customer_id: str = ""
    metric: str = ""
    quantity: float = 0.0
    metadata: dict[str, str] = Field(default_factory=dict)


class MeterEventBatch(BaseModel):
    """Batch of submitted meter events."""

    events: Sequence[MeterEvent] = Field(default_factory=list)
    accepted_count: int = 0


@runtime_checkable
class BillingMeterBackend(Protocol):
    """Read-only usage metering hook for future harness SaaS path."""

    def list_meter_events(self, *, customer_id: str, limit: int = 50) -> Sequence[MeterEvent]:
        """List recent meter events for a customer."""

    def submit_meter_event(
        self,
        *,
        customer_id: str,
        metric: str,
        quantity: float,
    ) -> MeterEvent:
        """Record a single usage meter event."""
