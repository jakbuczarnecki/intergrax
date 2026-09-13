"""Observation sink that projects into the platform RuntimeEvent spine (P1B)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.contracts.application_execution_stage_signal import (
    ApplicationExecutionCorrelation,
    ApplicationExecutionStageSignalEmitter,
    ApplicationExecutionStageSignalEmissionError,
    ApplicationExecutionStageSignalError,
)

from platform_proofs.scenarios.verified_product_identification.application.observability.contracts import (
    ProductIdentificationObservation,
)
from platform_proofs.scenarios.verified_product_identification.application.observability.platform_projection import (
    project_product_identification_observation,
)
from platform_proofs.scenarios.verified_product_identification.application.observability.ports import (
    ObservationSinkError,
    ProductIdentificationObservationSink,
)


@dataclass(frozen=True, slots=True)
class PlatformProjectingProductIdentificationObservationSink:
    """
    Decorator sink: preserve scenario-owned recording, then project to platform spine.

    ``execution_correlation`` must be fixed for the pipeline run. Scenario observations are
    recorded on ``inner`` first; correlation is validated before platform projection so
    central-spine events never attach to the wrong execution identity.
    """

    inner: ProductIdentificationObservationSink
    emitter: ApplicationExecutionStageSignalEmitter
    execution_correlation: ApplicationExecutionCorrelation

    def record(self, observation: ProductIdentificationObservation) -> None:
        self.inner.record(observation)
        if observation.run_id.value != self.execution_correlation.scenario_execution_correlation_id:
            raise ObservationSinkError(
                "observation run_id does not match execution correlation "
                "scenario_execution_correlation_id",
            )
        signal = project_product_identification_observation(observation)
        try:
            self.emitter.emit(signal, correlation=self.execution_correlation)
        except ApplicationExecutionStageSignalEmissionError as exc:
            raise ObservationSinkError(
                "platform application execution stage signal emission failed",
            ) from exc
        except ApplicationExecutionStageSignalError as exc:
            raise ObservationSinkError(str(exc)) from exc


@dataclass(frozen=True, slots=True)
class ChainedProductIdentificationObservationSink:
    """Fan-out to multiple scenario sinks without changing recorder semantics."""

    sinks: tuple[ProductIdentificationObservationSink, ...]

    def record(self, observation: ProductIdentificationObservation) -> None:
        for sink in self.sinks:
            sink.record(observation)
