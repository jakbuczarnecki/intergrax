# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.runtime.observability.extension_sdk import (
    PayloadSchemaRegistry,
    agent_signal_event_kind,
)
from external_contractor_adapter.signals.example_signal import MilestoneReachedPayloadV1


def register_signal_schemas() -> None:
    """Register agent domain signal kinds with the Harness event kind registry."""
    PayloadSchemaRegistry.register_runtime_extension(
        MilestoneReachedPayloadV1,
        event_kind=agent_signal_event_kind("external_contractor_adapter", "milestone_reached"),
    )
