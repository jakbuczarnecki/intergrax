# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.integrations.contracts.billing_meter import MeterEvent
from intergrax.tools.providers.billing.contracts import BillingListUsageInput, BillingRecordUsageInput
from intergrax.tools.providers.billing.service import billing_list_usage, billing_record_usage
from intergrax.tools.registry.wiring import ToolWiringContext

pytestmark = pytest.mark.unit


class FakeBillingMeter:
    def __init__(self) -> None:
        self.events: list[MeterEvent] = []

    def submit_meter_event(self, *, customer_id: str, metric: str, quantity: float) -> MeterEvent:
        event = MeterEvent(event_id=f"evt-{len(self.events)+1}", customer_id=customer_id, metric=metric, quantity=quantity)
        self.events.append(event)
        return event

    def list_meter_events(self, *, customer_id: str, limit: int = 50) -> list[MeterEvent]:
        return [item for item in self.events if item.customer_id == customer_id][:limit]


def test_billing_record_and_list_usage() -> None:
    backend = FakeBillingMeter()
    ctx = ToolWiringContext(billing_meter=backend)
    recorded = billing_record_usage(
        ctx,
        BillingRecordUsageInput(customer_id="cust-1", metric="tokens", quantity=12.5),
    )
    assert recorded.recorded is True
    listed = billing_list_usage(ctx, BillingListUsageInput(customer_id="cust-1"))
    assert listed.total == 1
    assert listed.events[0].metric == "tokens"
