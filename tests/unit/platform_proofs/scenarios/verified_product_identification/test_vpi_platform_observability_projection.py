"""VPI platform diagnostic projection tests (P1B)."""

from __future__ import annotations

import pytest

from intergrax.contracts.application_execution_stage_signal import (
    ApplicationExecutionStageSignal,
)
from intergrax.contracts.event_severity import EventSeverity
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.events.event_kind_registry import clear_event_kind_registry
from intergrax.runtime.events.runtime_event import RuntimeEventType
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
from platform_proofs.scenarios.verified_product_identification.application.observability.execution_correlation import (
    mint_lab_vpi_application_execution_correlation,
)
from platform_proofs.scenarios.verified_product_identification.application.observability.platform_projection import (
    project_product_identification_observation,
)
from platform_proofs.scenarios.verified_product_identification.application.observability.projection_sink import (
    PlatformProjectingProductIdentificationObservationSink,
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
