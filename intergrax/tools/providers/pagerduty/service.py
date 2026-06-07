# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.tools.providers.pagerduty.contracts import (
    PagerDutyAcknowledgeIncidentInput,
    PagerDutyAcknowledgeIncidentOutput,
    PagerDutyTriggerIncidentInput,
    PagerDutyTriggerIncidentOutput,
)
from intergrax.tools.providers.pagerduty.incident_channel import PagerDutyIncidentChannel
from intergrax.tools.registry.wiring import ToolWiringContext

PAGERDUTY_TRIGGER_INCIDENT_TOOL_ID = "pagerduty.trigger_incident"
PAGERDUTY_ACKNOWLEDGE_INCIDENT_TOOL_ID = "pagerduty.acknowledge_incident"


def _require_incident_channel(ctx: ToolWiringContext) -> PagerDutyIncidentChannel:
    channel = ctx.notification_channel
    if channel is None:
        raise RuntimeError("notification_channel_not_configured")
    if not isinstance(channel, PagerDutyIncidentChannel):
        raise RuntimeError("notification_channel_does_not_support_pagerduty_incidents")
    return channel


def pagerduty_trigger_incident(
    ctx: ToolWiringContext,
    params: PagerDutyTriggerIncidentInput,
) -> PagerDutyTriggerIncidentOutput:
    channel = _require_incident_channel(ctx)
    dedup_key = channel.trigger_incident(
        summary=params.summary.strip(),
        severity=params.severity,
        source=params.source,
        custom_details=params.custom_details,
        dedup_key=params.dedup_key or None,
    )
    return PagerDutyTriggerIncidentOutput(dedup_key=str(dedup_key), triggered=True)


def pagerduty_acknowledge_incident(
    ctx: ToolWiringContext,
    params: PagerDutyAcknowledgeIncidentInput,
) -> PagerDutyAcknowledgeIncidentOutput:
    channel = _require_incident_channel(ctx)
    dedup_key = params.dedup_key.strip()
    channel.acknowledge_incident(dedup_key=dedup_key, note=params.note or None)
    return PagerDutyAcknowledgeIncidentOutput(dedup_key=dedup_key, acknowledged=True)
