# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""
Public ERL reliability operator diagnostics contracts (ERL-DIAG-001A).

Package placement (OQ-1 DECIDED): ``intergrax.contracts.enterprise_reliability.diagnostics``
— ERL is the source of diagnostic facts; central diagnostics consumes these contracts
without creating a dependency cycle with ``intergrax.contracts.diagnostics`` persistence ports.
"""

from __future__ import annotations

from intergrax.contracts.enterprise_reliability.diagnostics.artifact_refs import (
    ReliabilityDiagnosticArtifactRefs,
)
from intergrax.contracts.enterprise_reliability.diagnostics.correlation import (
    ReliabilityDiagnosticCorrelation,
)
from intergrax.contracts.enterprise_reliability.diagnostics.emitter import (
    ExternalEffectReliabilityDiagnosticEmitter,
    NullExternalEffectReliabilityDiagnosticEmitter,
)
from intergrax.contracts.enterprise_reliability.diagnostics.observation import (
    SCHEMA_EXTERNAL_EFFECT_RELIABILITY_OBSERVATION_V1,
    ExternalEffectReliabilityObservation,
    ExternalEffectReliabilityObservationValidationError,
    MAX_RELIABILITY_DIAGNOSTIC_TRACE_REFS,
)
from intergrax.contracts.enterprise_reliability.diagnostics.taxonomy import (
    SCHEMA_AUTOMATION_SAFETY_HINT_V1,
    SCHEMA_EXTERNAL_EFFECT_RELIABILITY_SIGNAL_KIND_V1,
    AutomationSafetyHint,
    ExternalEffectReliabilitySignalKind,
)

__all__ = [
    "AutomationSafetyHint",
    "ExternalEffectReliabilityDiagnosticEmitter",
    "ExternalEffectReliabilityObservation",
    "ExternalEffectReliabilityObservationValidationError",
    "ExternalEffectReliabilitySignalKind",
    "MAX_RELIABILITY_DIAGNOSTIC_TRACE_REFS",
    "NullExternalEffectReliabilityDiagnosticEmitter",
    "ReliabilityDiagnosticArtifactRefs",
    "ReliabilityDiagnosticCorrelation",
    "SCHEMA_AUTOMATION_SAFETY_HINT_V1",
    "SCHEMA_EXTERNAL_EFFECT_RELIABILITY_OBSERVATION_V1",
    "SCHEMA_EXTERNAL_EFFECT_RELIABILITY_SIGNAL_KIND_V1",
]
