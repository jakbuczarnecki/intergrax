# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest
from pydantic import Field

from intergrax.contracts.execution_identity import mint_attempt_id, mint_execution_id, mint_run_id, mint_task_id
from intergrax.runtime.events.emit_context import EmitContext
from testing_support.runtime_events import emit_context_test_identity
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.events.event_catalog import EventCategory
from intergrax.runtime.events.payloads.base import RuntimeEventPayload
from intergrax.runtime.events.runtime_event import RuntimeEventType
from intergrax.runtime.events.signals import DomainSignalError, emit_domain_signal, emit_platform_event
from intergrax.runtime.events.payloads.canonical import ToolPayloadV1
from intergrax.runtime.events.event_kind_registry import clear_event_kind_registry
from intergrax.runtime.observability.extension_sdk import register_extension_runtime_payload

pytestmark = pytest.mark.gate


class _LegalClauseFlaggedV1(RuntimeEventPayload):
    schema_id = "agents.legal.clause_flagged.v1"
    clause_id: str = Field(...)
    score: float = Field(ge=0.0, le=1.0)

    def redact(self) -> _LegalClauseFlaggedV1:
        return self


@pytest.fixture(autouse=True)
def _register_legal_domain_kind() -> None:
    clear_event_kind_registry()
    register_extension_runtime_payload(
        _LegalClauseFlaggedV1,
        event_kind="agents.legal.clause_flagged",
    )
    yield
    clear_event_kind_registry()


@pytest.fixture
def emit_ctx() -> EmitContext:
    return emit_context_test_identity()


def test_emit_domain_signal_records_on_bus(emit_ctx: EmitContext) -> None:
    bus = RuntimeEventBus(record_history=True)
    ctx = emit_context_test_identity(
        task_id=emit_ctx.task_id,
        run_id=emit_ctx.run_id,
        attempt_id=emit_ctx.attempt_id,
        execution_id=emit_ctx.execution_id,
        tenant_id="tenant-a",
        bus=bus,
    )
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


def test_emit_domain_signal_redacts_in_production_mode(emit_ctx: EmitContext) -> None:
    class _SecretPayload(RuntimeEventPayload):
        schema_id = "agents.legal.secret.v1"
        secret: str = Field(...)

        def redact(self) -> _SecretPayload:
            return self.model_copy(update={"secret": "[REDACTED]"})

    register_extension_runtime_payload(
        _SecretPayload,
        event_kind="agents.legal.secret_flagged",
    )
    ctx = emit_context_test_identity(
        task_id=emit_ctx.task_id,
        run_id=emit_ctx.run_id,
        attempt_id=emit_ctx.attempt_id,
        execution_id=emit_ctx.execution_id,
        production_mode=True,
    )
    event = emit_domain_signal(
        ctx,
        kind="agents.legal.secret_flagged",
        payload=_SecretPayload(secret="top-secret"),
    )
    assert event.payload["data"]["secret"] == "[REDACTED]"


def test_emit_platform_event_uses_catalog_defaults(emit_ctx: EmitContext) -> None:
    bus = RuntimeEventBus(record_history=True)
    ctx = emit_context_test_identity(
        task_id=emit_ctx.task_id,
        run_id=emit_ctx.run_id,
        attempt_id=emit_ctx.attempt_id,
        execution_id=emit_ctx.execution_id,
        bus=bus,
    )
    event = emit_platform_event(
        ctx,
        event_type=RuntimeEventType.TOOL_COMPLETED,
        payload=ToolPayloadV1(tool_name="search", status="completed"),
    )
    assert event.event_kind == "tool_completed"
    assert event.event_category == EventCategory.TOOL
    assert len(bus.history) == 1


def test_emit_domain_signal_rejects_invalid_kind(emit_ctx: EmitContext) -> None:
    ctx = emit_ctx
    with pytest.raises(DomainSignalError):
        emit_domain_signal(
            ctx,
            kind="invalid_kind",
            payload=_LegalClauseFlaggedV1(clause_id="c-1", score=0.5),
        )
