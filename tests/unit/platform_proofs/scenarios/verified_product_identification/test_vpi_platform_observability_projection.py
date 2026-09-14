"""VPI platform diagnostic projection tests (P1B)."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from intergrax.contracts.application_execution_stage_signal import (
    ApplicationExecutionCorrelation,
    ApplicationExecutionStageSignal,
    ApplicationExecutionStageSignalEmissionError,
)
from intergrax.contracts.event_severity import EventSeverity
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.events.event_kind_registry import clear_event_kind_registry
from intergrax.runtime.events.runtime_event import RuntimeEventType
from intergrax.runtime.events.stores.memory_runtime_event_store import InMemoryRuntimeEventStore
from intergrax.runtime.observability.application_execution_stage_signal import (
    APPLICATION_EXECUTION_STAGE_OBSERVED_EVENT_KIND,
    RuntimeEventApplicationExecutionStageSignalEmitter,
    register_application_execution_stage_domain_signals,
)
from platform_proofs.scenarios.verified_product_identification.application.observability.contracts import (
    ProductIdentificationEventKind,
    ProductIdentificationInputOrigin,
    ProductIdentificationObservation,
    ProductIdentificationRunId,
    ProductIdentificationStage,
    QueryContextObservedPayload,
    TerminalObservedPayload,
)
from platform_proofs.scenarios.verified_product_identification.lab.execution_correlation_lab import (
    mint_lab_vpi_application_execution_correlation,
)
from platform_proofs.scenarios.verified_product_identification.application.observability.platform_projection import (
    project_product_identification_observation,
)
from platform_proofs.scenarios.verified_product_identification.application.observability.projection_sink import (
    PlatformProjectingProductIdentificationObservationSink,
)
from platform_proofs.scenarios.verified_product_identification.application.observability.ports import (
    ObservationSinkError,
    ProductIdentificationObservationSinkMode,
)
from platform_proofs.scenarios.verified_product_identification.application.observability.recorder import (
    ProductIdentificationObservationRecorder,
)
from platform_proofs.scenarios.verified_product_identification.application.observability.sinks import (
    InMemoryProductIdentificationObservationSink,
)
from platform_proofs.scenarios.verified_product_identification.application.domain import (
    ProductIdentifier,
    ProductIdentifierType,
)
from platform_proofs.scenarios.verified_product_identification.application.contracts.identification_context import (
    ProductIdentificationQueryContext,
)
from platform_proofs.scenarios.verified_product_identification.application.verification.contracts import (
    ProductIdentificationDecisionReasonCode,
    ProductIdentificationOutcome,
)

pytestmark = pytest.mark.gate


@pytest.fixture(autouse=True)
def _clear_kind_registry() -> None:
    clear_event_kind_registry()
    register_application_execution_stage_domain_signals()


def test_vpi_observation_projects_to_neutral_signal_without_domain_payload() -> None:
    run_id = ProductIdentificationRunId(value="660e8400-e29b-41d4-a716-446655440001")
    observation = ProductIdentificationObservation(
        run_id=run_id,
        sequence=2,
        stage=ProductIdentificationStage.QUERY_CONTEXT,
        kind=ProductIdentificationEventKind.QUERY_CONTEXT,
        payload=QueryContextObservedPayload(
            input_origin=ProductIdentificationInputOrigin.RAW_QUERY,
            query_context=ProductIdentificationQueryContext(
                requested_identifiers=(
                    ProductIdentifier(
                        identifier_type=ProductIdentifierType.MPN,
                        value="secret-sku-12345",
                    ),
                ),
            ),
        ),
    )
    signal = project_product_identification_observation(observation)
    assert isinstance(signal, ApplicationExecutionStageSignal)
    assert signal.scenario_execution_correlation_id == run_id.value
    assert signal.sequence == 2
    assert "secret" not in signal.summary.lower()
    assert "12345" not in signal.summary
    assert "mpn" not in signal.summary.lower()


def test_platform_projecting_sink_preserves_inner_observations_and_emits_runtime_events() -> None:
    bus = RuntimeEventBus(record_history=True)
    inner = InMemoryProductIdentificationObservationSink()
    run_id = ProductIdentificationRunId(value="770e8400-e29b-41d4-a716-446655440002")
    correlation = mint_lab_vpi_application_execution_correlation(
        tenant_id="tenant-vpi-lab",
        scenario_run_id=run_id,
    )
    sink = PlatformProjectingProductIdentificationObservationSink(
        inner=inner,
        emitter=RuntimeEventApplicationExecutionStageSignalEmitter(bus=bus, production_mode=True),
        execution_correlation=correlation,
    )
    observation = ProductIdentificationObservation(
        run_id=run_id,
        sequence=0,
        stage=ProductIdentificationStage.TERMINAL,
        kind=ProductIdentificationEventKind.TERMINAL,
        payload=TerminalObservedPayload(
            outcome=ProductIdentificationOutcome.VERIFIED,
            reason_code=ProductIdentificationDecisionReasonCode.UNIQUE_IDENTITY_SUPPORTED,
            verified_hypothesis_id="hyp-1",
            clarification_required=False,
        ),
    )
    sink.record(observation)
    assert inner.snapshot() == (observation,)
    assert len(bus.history) == 1
    runtime = bus.history[0]
    assert runtime.event_type is RuntimeEventType.DOMAIN_SIGNAL
    assert runtime.event_kind == APPLICATION_EXECUTION_STAGE_OBSERVED_EVENT_KIND
    assert runtime.correlation_id == run_id.value
    data = runtime.payload["data"]
    assert data["outcome_status"] == ProductIdentificationOutcome.VERIFIED.value
    assert data["diagnostic_code"] == ProductIdentificationDecisionReasonCode.UNIQUE_IDENTITY_SUPPORTED.value
    assert data["severity"] == EventSeverity.INFO.value


@dataclass(frozen=True, slots=True)
class _EmissionFailingStageSignalEmitter:
    def emit(
        self,
        signal: ApplicationExecutionStageSignal,
        *,
        correlation: ApplicationExecutionCorrelation,
    ) -> None:
        raise ApplicationExecutionStageSignalEmissionError("injected emission failure")


def test_correlation_mismatch_raises_observation_sink_error_without_runtime_event() -> None:
    bus = RuntimeEventBus(record_history=True)
    inner = InMemoryProductIdentificationObservationSink()
    configured_run = ProductIdentificationRunId(value="run-a")
    mismatched_run = ProductIdentificationRunId(value="run-b")
    correlation = mint_lab_vpi_application_execution_correlation(
        tenant_id="tenant-vpi-lab",
        scenario_run_id=configured_run,
    )
    sink = PlatformProjectingProductIdentificationObservationSink(
        inner=inner,
        emitter=RuntimeEventApplicationExecutionStageSignalEmitter(bus=bus, production_mode=True),
        execution_correlation=correlation,
    )
    observation = ProductIdentificationObservation(
        run_id=mismatched_run,
        sequence=0,
        stage=ProductIdentificationStage.TERMINAL,
        kind=ProductIdentificationEventKind.TERMINAL,
        payload=TerminalObservedPayload(
            outcome=ProductIdentificationOutcome.VERIFIED,
            reason_code=ProductIdentificationDecisionReasonCode.UNIQUE_IDENTITY_SUPPORTED,
            verified_hypothesis_id="hyp-1",
            clarification_required=False,
        ),
    )
    with pytest.raises(ObservationSinkError, match="does not match execution correlation"):
        sink.record(observation)
    assert inner.snapshot() == (observation,)
    assert not bus.history


def test_recorder_best_effort_continues_after_projection_emission_failure() -> None:
    inner = InMemoryProductIdentificationObservationSink()
    run_id = ProductIdentificationRunId(value="880e8400-e29b-41d4-a716-446655440003")
    correlation = mint_lab_vpi_application_execution_correlation(
        tenant_id="tenant-vpi-lab",
        scenario_run_id=run_id,
    )
    projecting_sink = PlatformProjectingProductIdentificationObservationSink(
        inner=inner,
        emitter=_EmissionFailingStageSignalEmitter(),
        execution_correlation=correlation,
    )
    recorder = ProductIdentificationObservationRecorder(
        run_id=run_id,
        sink=projecting_sink,
        sink_mode=ProductIdentificationObservationSinkMode.BEST_EFFORT,
    )
    recorder.record_payload(
        stage=ProductIdentificationStage.TERMINAL,
        payload=TerminalObservedPayload(
            outcome=ProductIdentificationOutcome.VERIFIED,
            reason_code=ProductIdentificationDecisionReasonCode.UNIQUE_IDENTITY_SUPPORTED,
            verified_hypothesis_id="hyp-1",
            clarification_required=False,
        ),
    )
    assert len(inner.snapshot()) == 1


class _MandatoryPersistenceFailingStore(InMemoryRuntimeEventStore):
    def append(self, event, *, tenant_id: str):
        raise OSError("simulated mandatory persistence backend outage")


def _terminal_payload() -> TerminalObservedPayload:
    return TerminalObservedPayload(
        outcome=ProductIdentificationOutcome.VERIFIED,
        reason_code=ProductIdentificationDecisionReasonCode.UNIQUE_IDENTITY_SUPPORTED,
        verified_hypothesis_id="hyp-1",
        clarification_required=False,
    )


def test_recorder_best_effort_continues_after_real_mandatory_persistence_failure() -> None:
    inner = InMemoryProductIdentificationObservationSink()
    run_id = ProductIdentificationRunId(value="aa0e8400-e29b-41d4-a716-446655440005")
    correlation = mint_lab_vpi_application_execution_correlation(
        tenant_id="tenant-vpi-lab",
        scenario_run_id=run_id,
    )
    bus = RuntimeEventBus(
        persistence=_MandatoryPersistenceFailingStore(),
        record_history=True,
    )
    projecting_sink = PlatformProjectingProductIdentificationObservationSink(
        inner=inner,
        emitter=RuntimeEventApplicationExecutionStageSignalEmitter(bus=bus, production_mode=True),
        execution_correlation=correlation,
    )
    recorder = ProductIdentificationObservationRecorder(
        run_id=run_id,
        sink=projecting_sink,
        sink_mode=ProductIdentificationObservationSinkMode.BEST_EFFORT,
    )
    recorder.record_payload(
        stage=ProductIdentificationStage.TERMINAL,
        payload=_terminal_payload(),
    )
    assert len(inner.snapshot()) == 1
    assert not bus.history


def test_recorder_required_propagates_after_real_mandatory_persistence_failure() -> None:
    inner = InMemoryProductIdentificationObservationSink()
    run_id = ProductIdentificationRunId(value="ab0e8400-e29b-41d4-a716-446655440006")
    correlation = mint_lab_vpi_application_execution_correlation(
        tenant_id="tenant-vpi-lab",
        scenario_run_id=run_id,
    )
    bus = RuntimeEventBus(
        persistence=_MandatoryPersistenceFailingStore(),
        record_history=True,
    )
    projecting_sink = PlatformProjectingProductIdentificationObservationSink(
        inner=inner,
        emitter=RuntimeEventApplicationExecutionStageSignalEmitter(bus=bus, production_mode=True),
        execution_correlation=correlation,
    )
    recorder = ProductIdentificationObservationRecorder(
        run_id=run_id,
        sink=projecting_sink,
        sink_mode=ProductIdentificationObservationSinkMode.REQUIRED,
    )
    with pytest.raises(ObservationSinkError, match="platform application execution stage signal"):
        recorder.record_payload(
            stage=ProductIdentificationStage.TERMINAL,
            payload=_terminal_payload(),
        )
    assert len(inner.snapshot()) == 1
    assert not bus.history


def test_recorder_required_propagates_projection_emission_failure() -> None:
    inner = InMemoryProductIdentificationObservationSink()
    run_id = ProductIdentificationRunId(value="990e8400-e29b-41d4-a716-446655440004")
    correlation = mint_lab_vpi_application_execution_correlation(
        tenant_id="tenant-vpi-lab",
        scenario_run_id=run_id,
    )
    projecting_sink = PlatformProjectingProductIdentificationObservationSink(
        inner=inner,
        emitter=_EmissionFailingStageSignalEmitter(),
        execution_correlation=correlation,
    )
    recorder = ProductIdentificationObservationRecorder(
        run_id=run_id,
        sink=projecting_sink,
        sink_mode=ProductIdentificationObservationSinkMode.REQUIRED,
    )
    with pytest.raises(ObservationSinkError, match="platform application execution stage signal"):
        recorder.record_payload(
            stage=ProductIdentificationStage.TERMINAL,
            payload=TerminalObservedPayload(
                outcome=ProductIdentificationOutcome.VERIFIED,
                reason_code=ProductIdentificationDecisionReasonCode.UNIQUE_IDENTITY_SUPPORTED,
                verified_hypothesis_id="hyp-1",
                clarification_required=False,
            ),
        )
