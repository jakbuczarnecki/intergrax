# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from pydantic import Field

from intergrax.runtime.events.payloads.base import RuntimeEventPayload
from intergrax.runtime.observability.extension_sdk import agent_signal_schema_id


class MilestoneReachedPayloadV1(RuntimeEventPayload):
    """Example operator-visible domain signal — replace with product semantics."""

    schema_id = agent_signal_schema_id("external_contractor_adapter", "milestone_reached")
    milestone: str = Field(min_length=1)
    detail: str = ""

    def redact(self) -> MilestoneReachedPayloadV1:
        return self
