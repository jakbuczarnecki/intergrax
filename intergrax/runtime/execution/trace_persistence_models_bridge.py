# © Artur Czarnecki. All rights reserved.

from typing import Any

from intergrax.runtime.nexus.artifacts.models import TraceArtifactRef
from intergrax.runtime.nexus.tracing.adapters.core_llm_call_recorded import CoreLLMCallRecordedDiagV1
from intergrax.runtime.nexus.tracing.execution.completion_alignment import CompletionAlignmentDiagV1
from intergrax.runtime.nexus.tracing.execution.evaluator_model_attempt import (
    EvaluatorModelAttemptDiagV1,
)
from intergrax.runtime.nexus.tracing.execution.reconciliation_phase import (
    ReconciliationPhaseDiagV1,
    ReconciliationPhaseValue,
)
from intergrax.runtime.nexus.tracing.parser_trace_flush import export_parser_traces_from_events
from intergrax.runtime.nexus.tracing.persisted_trace_codec import persisted_trace_event_to_serialized
from intergrax.runtime.nexus.tracing.persistence_models import SerializedTraceEvent

__all__ = [
    "CompletionAlignmentDiagV1",
    "CoreLLMCallRecordedDiagV1",
    "EvaluatorModelAttemptDiagV1",
    "ReconciliationPhaseDiagV1",
    "ReconciliationPhaseValue",
    "SerializedTraceEvent",
    "TraceArtifactRef",
    "export_parser_traces_from_events",
    "persisted_trace_event_to_serialized",
]


def __getattr__(name: str) -> Any:
    """COMPAT-X shim: ``ArtifactRef`` was an alias of ``TraceArtifactRef`` (PDC-D8-R018).

    The alias is no longer bound at module level; existing imports keep resolving.
    """
    if name == "ArtifactRef":
        return TraceArtifactRef
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
