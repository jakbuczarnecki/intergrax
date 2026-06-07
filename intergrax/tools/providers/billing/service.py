# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.integrations.contracts.billing_meter import BillingMeterBackend
from intergrax.tools.providers.billing.contracts import (
    BillingListUsageInput,
    BillingListUsageOutput,
    BillingMeterEventOutput,
    BillingRecordUsageInput,
    BillingRecordUsageOutput,
)
from intergrax.tools.registry.wiring import ToolWiringContext

BILLING_RECORD_USAGE_TOOL_ID = "billing.record_usage"
BILLING_LIST_USAGE_TOOL_ID = "billing.list_usage"


def _require_billing(ctx: ToolWiringContext) -> BillingMeterBackend:
    backend = ctx.billing_meter
    if backend is None:
        raise RuntimeError("billing_meter_not_configured")
    return backend


def billing_record_usage(ctx: ToolWiringContext, params: BillingRecordUsageInput) -> BillingRecordUsageOutput:
    event = _require_billing(ctx).submit_meter_event(
        customer_id=params.customer_id.strip(),
        metric=params.metric.strip(),
        quantity=params.quantity,
    )
    return BillingRecordUsageOutput(
        event_id=event.event_id,
        customer_id=event.customer_id,
        metric=event.metric,
        quantity=event.quantity,
    )


def billing_list_usage(ctx: ToolWiringContext, params: BillingListUsageInput) -> BillingListUsageOutput:
    events = [
        BillingMeterEventOutput(
            event_id=item.event_id,
            customer_id=item.customer_id,
            metric=item.metric,
            quantity=item.quantity,
        )
        for item in _require_billing(ctx).list_meter_events(
            customer_id=params.customer_id.strip(),
            limit=params.limit,
        )
    ]
    return BillingListUsageOutput(
        customer_id=params.customer_id.strip(),
        events=events,
        total=len(events),
    )
