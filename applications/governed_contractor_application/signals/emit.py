# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.runtime.events.emit_context import EmitContext
from intergrax.runtime.events.runtime_event import RuntimeEvent
from intergrax.runtime.events.signals import emit_domain_signal
from intergrax.runtime.observability.extension_sdk import application_signal_event_kind
from applications.governed_contractor_application.signals.example_signal import HostReadyPayloadV1


def emit_host_ready(ctx: EmitContext, *, phase: str) -> RuntimeEvent:
    """Emit a typed domain signal when the host reaches a lifecycle milestone."""
    return emit_domain_signal(
        ctx,
        kind=application_signal_event_kind("governed_contractor", "host_ready"),
        payload=HostReadyPayloadV1(phase=phase),
    )
