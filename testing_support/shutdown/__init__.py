# © Artur Czarnecki. All rights reserved.

"""Reference graceful-shutdown surface for EE-B4-B certification (not runtime authority)."""

from testing_support.shutdown.models import (
    ReferenceRootAdmissionDecision,
    ReferenceShutdownFailureKind,
    ReferenceShutdownTerminalOutcome,
)
from testing_support.shutdown.reference_lifecycle import (
    ReferenceExecutionShutdownLifecycle,
)
from testing_support.shutdown.ports import (
    InMemoryFinalStateStore,
    RecordingMandatoryEvidenceFlush,
    RecordingObservabilityExporter,
)

__all__ = [
    "InMemoryFinalStateStore",
    "RecordingMandatoryEvidenceFlush",
    "RecordingObservabilityExporter",
    "ReferenceExecutionShutdownLifecycle",
    "ReferenceRootAdmissionDecision",
    "ReferenceShutdownFailureKind",
    "ReferenceShutdownTerminalOutcome",
]
