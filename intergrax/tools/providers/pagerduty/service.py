# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.tools.providers.pagerduty.contracts import (
    PagerDutyAcknowledgeIncidentInput,
    PagerDutyAcknowledgeIncidentOutput,
    PagerDutyTriggerIncidentInput,
    PagerDutyTriggerIncidentOutput,
)
from intergrax.tools.registry.wiring import ToolWiringContext

PAGERDUTY_TRIGGER_INCIDENT_TOOL_ID = "pagerduty.trigger_incident"
PAGERDUTY_ACKNOWLEDGE_INCIDENT_TOOL_ID = "pagerduty.acknowledge_incident"


def pagerduty_trigger_incident(
    ctx: ToolWiringContext,
    params: PagerDutyTriggerIncidentInput,
) -> PagerDutyTriggerIncidentOutput:
    channel = ctx.notification_channel
    if channel is None:
        raise RuntimeError("notification_channel_not_configured")
    trigger = getattr(channel, "trigger_incident", None)
    if trigger is None:
        raise RuntimeError("notification_channel_does_not_support_incident_trigger")
    dedup_key = trigger(
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
    channel = ctx.notification_channel
    if channel is None:
        raise RuntimeError("notification_channel_not_configured")
    acknowledge = getattr(channel, "acknowledge_incident", None)
    if acknowledge is None:
        raise RuntimeError("notification_channel_does_not_support_incident_acknowledge")
    dedup_key = params.dedup_key.strip()
    acknowledge(dedup_key=dedup_key, note=params.note or None)
    return PagerDutyAcknowledgeIncidentOutput(dedup_key=dedup_key, acknowledged=True)
