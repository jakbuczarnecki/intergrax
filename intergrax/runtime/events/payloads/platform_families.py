# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Platform-owned domain signal payload families (OBS-DIAG-EC2)."""

from __future__ import annotations

from intergrax.runtime.events.payloads.base import RuntimeEventPayload


class DiagnosticSubsystemFailurePayloadV1(RuntimeEventPayload):
    """Emitted when terminal execution diagnostics fail to persist (DIAG-FOUNDATION-3)."""

    schema_id = "platform.diagnostic.subsystem_failure.v1"

    error_type: str = ""
    source: str = ""


__all__ = ["DiagnosticSubsystemFailurePayloadV1"]
