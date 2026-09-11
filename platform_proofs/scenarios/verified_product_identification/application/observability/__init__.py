"""VPI application production observability."""

from platform_proofs.scenarios.verified_product_identification.application.observability.contracts import (
    ClarificationObservedPayload,
    FusionObservedPayload,
    IdentityEvaluationObservedPayload,
    IdentityHypothesesObservedPayload,
    ProductIdentificationEventKind,
    ProductIdentificationInputOrigin,
    ProductIdentificationObservation,
    ProductIdentificationRunId,
    ProductIdentificationStage,
    QueryContextObservedPayload,
    RetrievalChannelObservedPayload,
    StageFailureObservedPayload,
    StageTimingObservedPayload,
    TerminalObservedPayload,
    VerificationObservedPayload,
)
from platform_proofs.scenarios.verified_product_identification.application.observability.ports import (
    MonotonicClockPort,
    ObservationSinkError,
    ProductIdentificationObservationSink,
    ProductIdentificationObservationSinkMode,
)
from platform_proofs.scenarios.verified_product_identification.application.observability.recorder import (
    ProductIdentificationObservationRecorder,
)
from platform_proofs.scenarios.verified_product_identification.application.observability.sinks import (
    FailingProductIdentificationObservationSink,
    InMemoryProductIdentificationObservationSink,
    NoOpProductIdentificationObservationSink,
)

__all__ = (
    "ClarificationObservedPayload",
    "FailingProductIdentificationObservationSink",
    "FusionObservedPayload",
    "IdentityEvaluationObservedPayload",
    "IdentityHypothesesObservedPayload",
    "InMemoryProductIdentificationObservationSink",
    "MonotonicClockPort",
    "NoOpProductIdentificationObservationSink",
    "ObservationSinkError",
    "ProductIdentificationEventKind",
    "ProductIdentificationInputOrigin",
    "ProductIdentificationObservation",
    "ProductIdentificationObservationRecorder",
    "ProductIdentificationObservationSink",
    "ProductIdentificationObservationSinkMode",
    "ProductIdentificationRunId",
    "ProductIdentificationStage",
    "QueryContextObservedPayload",
    "RetrievalChannelObservedPayload",
    "StageFailureObservedPayload",
    "StageTimingObservedPayload",
    "TerminalObservedPayload",
    "VerificationObservedPayload",
)
