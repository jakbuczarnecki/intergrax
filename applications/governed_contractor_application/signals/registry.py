# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.runtime.observability.extension_sdk import (
    PayloadSchemaRegistry,
    application_signal_event_kind,
)
from applications.governed_contractor_application.signals.example_signal import HostReadyPayloadV1


def register_signal_schemas() -> None:
    """Register application domain signal kinds with the Harness event kind registry."""
    PayloadSchemaRegistry.register_runtime_extension(
        HostReadyPayloadV1,
        event_kind=application_signal_event_kind("governed_contractor", "host_ready"),
    )
