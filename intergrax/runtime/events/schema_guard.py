# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Runtime event schema validation (§42.29)."""

from __future__ import annotations

from intergrax.runtime.events.phase_coverage import phase_for_event
from intergrax.runtime.events.runtime_event import RuntimeEvent
from intergrax.runtime.schema.registry import validate_schema_version


class RuntimeEventSchemaError(ValueError):
    """Raised when a runtime event violates canonical schema or phase mapping."""


def assert_runtime_event_schema(event: RuntimeEvent) -> None:
    if not validate_schema_version("runtime_event", event.schema_version):
        raise RuntimeEventSchemaError(
            f"unsupported runtime_event schema_version={event.schema_version!r}"
        )
    expected_phase = phase_for_event(event.event_type)
    if expected_phase is not None and event.phase != expected_phase:
        raise RuntimeEventSchemaError(
            f"phase mismatch for {event.event_type.value}: "
            f"expected {expected_phase.value}, got {event.phase.value}"
        )
