# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest
from pydantic import Field

from intergrax.runtime.events.emit_context import EmitContext
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.events.event_catalog import EventCategory
from intergrax.runtime.events.payloads.base import RuntimeEventPayload
from intergrax.runtime.events.runtime_event import RuntimeEventType
from intergrax.runtime.events.signals import DomainSignalError, emit_domain_signal, emit_platform_event
from intergrax.runtime.events.payloads.canonical import ToolPayloadV1
from intergrax.runtime.observability.extension_sdk import register_extension_runtime_payload

pytestmark = pytest.mark.gate


class _LegalClauseFlaggedV1(RuntimeEventPayload):
    schema_id = "agents.legal.clause_flagged.v1"
    clause_id: str = Field(...)
    score: float = Field(ge=0.0, le=1.0)

    def redact(self) -> _LegalClauseFlaggedV1:
        return self


register_extension_runtime_payload(_LegalClauseFlaggedV1)


def test_emit_domain_signal_records_on_bus() -> None:
    bus = RuntimeEventBus(record_history=True)
    ctx = EmitContext(task_id="task-1", run_id="run-1", tenant_id="tenant-a", bus=bus)
    event = emit_domain_signal(
        ctx,
        kind="agents.legal.clause_flagged",
        payload=_LegalClauseFlaggedV1(clause_id="c-1", score=0.9),
    )
    assert event.event_type == RuntimeEventType.DOMAIN_SIGNAL
    assert event.event_kind == "agents.legal.clause_flagged"
    assert event.event_category == EventCategory.AGENT
    assert bus.history[-1].event_id == event.event_id
    assert event.payload["payload_schema_id"] == _LegalClauseFlaggedV1.schema_id


def test_emit_domain_signal_redacts_in_production_mode() -> None:
    class _SecretPayload(RuntimeEventPayload):
        schema_id = "agents.legal.secret.v1"
        secret: str = Field(...)

        def redact(self) -> _SecretPayload:
            return self.model_copy(update={"secret": "[REDACTED]"})

    register_extension_runtime_payload(_SecretPayload)
    ctx = EmitContext(
        task_id="task-1",
        run_id="run-1",
        production_mode=True,
    )
    event = emit_domain_signal(
        ctx,
        kind="agents.legal.secret_flagged",
        payload=_SecretPayload(secret="top-secret"),
    )
    assert event.payload["data"]["secret"] == "[REDACTED]"


def test_emit_platform_event_uses_catalog_defaults() -> None:
    bus = RuntimeEventBus(record_history=True)
    ctx = EmitContext(task_id="task-1", run_id="run-1", bus=bus)
    event = emit_platform_event(
        ctx,
        event_type=RuntimeEventType.TOOL_COMPLETED,
        payload=ToolPayloadV1(tool_name="search", status="completed"),
    )
    assert event.event_kind == "tool_completed"
    assert event.event_category == EventCategory.TOOL
    assert len(bus.history) == 1


def test_emit_domain_signal_rejects_invalid_kind() -> None:
    ctx = EmitContext(task_id="task-1", run_id="run-1")
    with pytest.raises(DomainSignalError):
        emit_domain_signal(
            ctx,
            kind="invalid_kind",
            payload=_LegalClauseFlaggedV1(clause_id="c-1", score=0.5),
        )
