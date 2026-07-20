# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.runtime.events.emit_context import EmitContext
from intergrax.runtime.events.runtime_event import RuntimeEvent
from intergrax.runtime.events.signals import emit_domain_signal
from intergrax.runtime.observability.extension_sdk import agent_signal_event_kind
from external_contractor_adapter.signals.example_signal import MilestoneReachedPayloadV1


def emit_milestone_reached(
    ctx: EmitContext,
    *,
    milestone: str,
    detail: str = "",
) -> RuntimeEvent:
    """Emit a typed domain signal for operator-visible agent milestones."""
    return emit_domain_signal(
        ctx,
        kind=agent_signal_event_kind("external_contractor_adapter", "milestone_reached"),
        payload=MilestoneReachedPayloadV1(milestone=milestone, detail=detail),
    )
