# © Artur Czarnecki. All rights reserved.

"""Emit neutral application stage signals on the DOMAIN_SIGNAL / RuntimeEvent spine."""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import model_validator

from intergrax.contracts.application_execution_stage_signal import (
    ApplicationExecutionCorrelation,
    ApplicationExecutionStageSignal,
    ApplicationExecutionStageSignalError,
)
from intergrax.contracts.event_severity import EventSeverity
from intergrax.runtime.events.emit_context import EmitContext
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.events.event_kind_registry import register_event_kind
from intergrax.runtime.events.payload_registry import register_payload_schema
from intergrax.runtime.events.payloads.base import RuntimeEventPayload
from intergrax.runtime.events.runtime_event import RuntimeEvent
from intergrax.runtime.events.signals import emit_domain_signal

APPLICATION_EXECUTION_STAGE_SIGNAL_PAYLOAD_SCHEMA_ID = (
    "applications.execution_stage.observed.v1"
)
APPLICATION_EXECUTION_STAGE_OBSERVED_EVENT_KIND = "applications.execution_stage.observed"


class ApplicationExecutionStageSignalPayloadV1(RuntimeEventPayload):
    """Typed, redaction-safe body for neutral application stage observations."""

    schema_id = APPLICATION_EXECUTION_STAGE_SIGNAL_PAYLOAD_SCHEMA_ID

    application_slug: str
    scenario_execution_correlation_id: str
    sequence: int
    stage_id: str
    event_category: str
    severity: str
    summary: str
    outcome_status: str | None = None
    diagnostic_code: str | None = None

    @model_validator(mode="after")
    def _validate_shape(self) -> ApplicationExecutionStageSignalPayloadV1:
        ApplicationExecutionStageSignal(
            application_slug=self.application_slug,
            scenario_execution_correlation_id=self.scenario_execution_correlation_id,
            sequence=self.sequence,
            stage_id=self.stage_id,
            event_category=self.event_category,
            severity=EventSeverity(self.severity),
            summary=self.summary,
            outcome_status=self.outcome_status,
            diagnostic_code=self.diagnostic_code,
        )
        return self

    def redact(self) -> ApplicationExecutionStageSignalPayloadV1:
        return self


def register_application_execution_stage_domain_signals() -> None:
    """Register payload schema and event kind (idempotent)."""
    register_payload_schema(ApplicationExecutionStageSignalPayloadV1, extension=True)
    register_event_kind(
        APPLICATION_EXECUTION_STAGE_OBSERVED_EVENT_KIND,
        APPLICATION_EXECUTION_STAGE_SIGNAL_PAYLOAD_SCHEMA_ID,
    )


def _payload_from_signal(signal: ApplicationExecutionStageSignal) -> ApplicationExecutionStageSignalPayloadV1:
    return ApplicationExecutionStageSignalPayloadV1(
        application_slug=signal.application_slug,
        scenario_execution_correlation_id=signal.scenario_execution_correlation_id,
        sequence=signal.sequence,
        stage_id=signal.stage_id,
        event_category=signal.event_category,
        severity=signal.severity.value,
        summary=signal.summary,
        outcome_status=signal.outcome_status,
        diagnostic_code=signal.diagnostic_code,
    )


def emit_application_execution_stage_signal(
    bus: RuntimeEventBus,
    *,
    correlation: ApplicationExecutionCorrelation,
    signal: ApplicationExecutionStageSignal,
    production_mode: bool = False,
) -> RuntimeEvent:
    """Project one neutral stage signal onto the canonical runtime event bus."""
    register_application_execution_stage_domain_signals()
    if signal.scenario_execution_correlation_id != correlation.scenario_execution_correlation_id:
        raise ApplicationExecutionStageSignalError(
            "signal scenario_execution_correlation_id must match correlation bundle",
        )
    ctx = EmitContext(
        task_id=correlation.task_id,
        run_id=correlation.run_id,
        attempt_id=correlation.attempt_id,
        execution_id=correlation.execution_id,
        tenant_id=correlation.tenant_id,
        correlation_id=correlation.scenario_execution_correlation_id,
        bus=bus,
        production_mode=production_mode,
    )
    return emit_domain_signal(
        ctx,
        kind=APPLICATION_EXECUTION_STAGE_OBSERVED_EVENT_KIND,
        payload=_payload_from_signal(signal),
        severity=signal.severity,
    )


@dataclass(frozen=True, slots=True)
class RuntimeEventApplicationExecutionStageSignalEmitter:
    """Default platform emitter over an injected ``RuntimeEventBus``."""

    bus: RuntimeEventBus
    production_mode: bool = False

    def emit(
        self,
        signal: ApplicationExecutionStageSignal,
        *,
        correlation: ApplicationExecutionCorrelation,
    ) -> None:
        emit_application_execution_stage_signal(
            self.bus,
            correlation=correlation,
            signal=signal,
            production_mode=self.production_mode,
        )


__all__ = [
    "APPLICATION_EXECUTION_STAGE_OBSERVED_EVENT_KIND",
    "APPLICATION_EXECUTION_STAGE_SIGNAL_PAYLOAD_SCHEMA_ID",
    "ApplicationExecutionStageSignalPayloadV1",
    "RuntimeEventApplicationExecutionStageSignalEmitter",
    "emit_application_execution_stage_signal",
    "register_application_execution_stage_domain_signals",
]
