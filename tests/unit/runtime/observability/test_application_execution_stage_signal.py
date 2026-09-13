# © Artur Czarnecki. All rights reserved.

"""Tests for neutral application execution stage signal projection."""

from __future__ import annotations

import pytest

from intergrax.contracts.application_execution_stage_signal import (
    ApplicationExecutionCorrelation,
    ApplicationExecutionStageSignal,
    ApplicationExecutionStageSignalEmissionError,
    ApplicationExecutionStageSignalError,
)
from intergrax.contracts.event_severity import EventSeverity
from intergrax.contracts.execution_identity import (
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
)
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.events.event_kind_registry import clear_event_kind_registry
from intergrax.runtime.events.runtime_event import RuntimeEventType
from intergrax.runtime.observability.application_execution_stage_signal import (
    APPLICATION_EXECUTION_STAGE_OBSERVED_EVENT_KIND,
    APPLICATION_EXECUTION_STAGE_SIGNAL_PAYLOAD_SCHEMA_ID,
    RuntimeEventApplicationExecutionStageSignalEmitter,
    emit_application_execution_stage_signal,
    register_application_execution_stage_domain_signals,
)
from intergrax.runtime.events.event_kind import DomainSignalError

pytestmark = pytest.mark.gate


@pytest.fixture(autouse=True)
def _clear_kind_registry() -> None:
    clear_event_kind_registry()
    register_application_execution_stage_domain_signals()


def test_emit_application_execution_stage_signal_on_bus() -> None:
    bus = RuntimeEventBus(record_history=True)
    correlation = ApplicationExecutionCorrelation(
        tenant_id="tenant-vpi",
        task_id=mint_task_id(),
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
        execution_id=mint_execution_id(),
        scenario_execution_correlation_id="550e8400-e29b-41d4-a716-446655440000",
    )
    signal = ApplicationExecutionStageSignal(
        application_slug="verified_product_identification",
        scenario_execution_correlation_id=correlation.scenario_execution_correlation_id,
        sequence=0,
        stage_id="retrieval",
        event_category="retrieval_channel",
        severity=EventSeverity.INFO,
        summary="retrieval channel=lexical status=succeeded candidates=3",
        outcome_status=None,
        diagnostic_code=None,
    )
    event = emit_application_execution_stage_signal(
        bus,
        correlation=correlation,
        signal=signal,
        production_mode=True,
    )
    assert event.event_type is RuntimeEventType.DOMAIN_SIGNAL
    assert event.event_kind == APPLICATION_EXECUTION_STAGE_OBSERVED_EVENT_KIND
    envelope = event.payload
    assert envelope["payload_schema_id"] == APPLICATION_EXECUTION_STAGE_SIGNAL_PAYLOAD_SCHEMA_ID
    data = envelope["data"]
    assert data["application_slug"] == "verified_product_identification"
    assert data["sequence"] == 0
    assert "gtin" not in data["summary"].lower()
    assert bus.history[-1].event_id == event.event_id


def test_platform_observability_module_has_no_vpi_imports() -> None:
    import intergrax.runtime.observability.application_execution_stage_signal as module

    source_path = module.__file__
    assert source_path is not None
    text = open(source_path, encoding="utf-8").read()
    assert "verified_product_identification" not in text
    assert "platform_proofs" not in text


def test_runtime_emitter_translates_domain_signal_error_to_public_emission_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raise_domain_signal_error(*_args: object, **_kwargs: object) -> None:
        raise DomainSignalError("injected domain signal failure")

    monkeypatch.setattr(
        "intergrax.runtime.observability.application_execution_stage_signal.emit_domain_signal",
        _raise_domain_signal_error,
    )
    bus = RuntimeEventBus(record_history=True)
    correlation = ApplicationExecutionCorrelation(
        tenant_id="tenant-vpi",
        task_id=mint_task_id(),
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
        execution_id=mint_execution_id(),
        scenario_execution_correlation_id="550e8400-e29b-41d4-a716-446655440000",
    )
    signal = ApplicationExecutionStageSignal(
        application_slug="verified_product_identification",
        scenario_execution_correlation_id=correlation.scenario_execution_correlation_id,
        sequence=0,
        stage_id="retrieval",
        event_category="retrieval_channel",
        severity=EventSeverity.INFO,
        summary="retrieval channel=lexical status=succeeded candidates=3",
    )
    with pytest.raises(ApplicationExecutionStageSignalEmissionError) as raised:
        RuntimeEventApplicationExecutionStageSignalEmitter(bus=bus).emit(
            signal,
            correlation=correlation,
        )
    assert isinstance(raised.value.__cause__, DomainSignalError)
    assert not bus.history


def test_signal_correlation_mismatch_raises_contract_error_not_emission_error() -> None:
    bus = RuntimeEventBus(record_history=True)
    correlation = ApplicationExecutionCorrelation(
        tenant_id="tenant-vpi",
        task_id=mint_task_id(),
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
        execution_id=mint_execution_id(),
        scenario_execution_correlation_id="550e8400-e29b-41d4-a716-446655440000",
    )
    signal = ApplicationExecutionStageSignal(
        application_slug="verified_product_identification",
        scenario_execution_correlation_id="660e8400-e29b-41d4-a716-446655440099",
        sequence=0,
        stage_id="retrieval",
        event_category="retrieval_channel",
        severity=EventSeverity.INFO,
        summary="retrieval channel=lexical status=succeeded candidates=3",
    )
    with pytest.raises(ApplicationExecutionStageSignalError, match="must match correlation"):
        emit_application_execution_stage_signal(bus, correlation=correlation, signal=signal)
    assert not bus.history
