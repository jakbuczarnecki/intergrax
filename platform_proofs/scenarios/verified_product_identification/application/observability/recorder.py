"""Sequence assignment and sink-mode policy for production observations."""

from __future__ import annotations

from dataclasses import dataclass, field

from platform_proofs.scenarios.verified_product_identification.application.observability.contracts import (
    ObservationPayload,
    ProductIdentificationEventKind,
    ProductIdentificationObservation,
    ProductIdentificationRunId,
    ProductIdentificationStage,
    _KIND_FOR_PAYLOAD,
    _STAGE_FOR_KIND,
)
from platform_proofs.scenarios.verified_product_identification.application.observability.ports import (
    ObservationSinkError,
    ProductIdentificationObservationSink,
    ProductIdentificationObservationSinkMode,
)


@dataclass
class ProductIdentificationObservationRecorder:
    run_id: ProductIdentificationRunId
    sink: ProductIdentificationObservationSink
    sink_mode: ProductIdentificationObservationSinkMode = (
        ProductIdentificationObservationSinkMode.BEST_EFFORT
    )
    _next_sequence: int = field(default=0, init=False)

    def record_payload(
        self,
        *,
        stage: ProductIdentificationStage,
        payload: ObservationPayload,
    ) -> None:
        kind = _KIND_FOR_PAYLOAD[type(payload)]
        if kind is ProductIdentificationEventKind.STAGE_TIMING:
            resolved_stage = stage
        else:
            resolved_stage = _STAGE_FOR_KIND.get(kind, stage)
        observation = ProductIdentificationObservation(
            run_id=self.run_id,
            sequence=self._next_sequence,
            stage=resolved_stage,
            kind=kind,
            payload=payload,
        )
        self._next_sequence += 1
        try:
            self.sink.record(observation)
        except ObservationSinkError:
            if self.sink_mode is ProductIdentificationObservationSinkMode.REQUIRED:
                raise
            return
