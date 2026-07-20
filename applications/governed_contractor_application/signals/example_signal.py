# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from pydantic import Field

from intergrax.runtime.events.payloads.base import RuntimeEventPayload
from intergrax.runtime.observability.extension_sdk import application_signal_schema_id


class HostReadyPayloadV1(RuntimeEventPayload):
    """Example operator-visible host signal — replace with product semantics."""

    schema_id = application_signal_schema_id("governed_contractor", "host_ready")
    phase: str = Field(min_length=1)

    def redact(self) -> HostReadyPayloadV1:
        return self
