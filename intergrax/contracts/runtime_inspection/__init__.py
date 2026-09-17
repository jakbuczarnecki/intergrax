# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Canonical cross-domain runtime inspection contracts (INSPECT-01-A)."""

from intergrax.contracts.runtime_inspection.completeness import RuntimeInspectionCompleteness
from intergrax.contracts.runtime_inspection.errors import (
    RuntimeInspectionError,
    RuntimeInspectionErrorCode,
    RuntimeInspectionNotFoundError,
    RuntimeInspectionTenantBoundaryError,
)
from intergrax.contracts.runtime_inspection.failures import (
    RuntimeInspectionSourceFailure,
    RuntimeInspectionSourceFailureCode,
)
from intergrax.contracts.runtime_inspection.query import (
    DEFAULT_RUNTIME_INSPECTION_TIMELINE_LIMIT,
    MAX_RUNTIME_INSPECTION_TIMELINE_LIMIT,
    RuntimeInspectionQuery,
)
from intergrax.contracts.runtime_inspection.read_port import RuntimeInspectionReadPort
from intergrax.contracts.runtime_inspection.sections import (
    RuntimeInspectionDiagnosticFinding,
    RuntimeInspectionDiagnosticSection,
    RuntimeInspectionEvidenceReference,
    RuntimeInspectionEvidenceSection,
    RuntimeInspectionExecutionStateSection,
    RuntimeInspectionIdentitySection,
    RuntimeInspectionTimelineDomain,
    RuntimeInspectionTimelineEntry,
    RuntimeInspectionTimelineSection,
)
from intergrax.contracts.runtime_inspection.snapshot import RuntimeInspectionSnapshot
from intergrax.contracts.runtime_inspection.sources import (
    RuntimeInspectionDiagnosticReadPort,
    RuntimeInspectionEvidenceReadPort,
    RuntimeInspectionExecutionFactsReader,
    RuntimeInspectionExecutionScope,
    RuntimeInspectionExecutionScopeReader,
    RuntimeInspectionScopeLookupOutcome,
    RuntimeInspectionScopeLookupResult,
)

__all__ = [
    "DEFAULT_RUNTIME_INSPECTION_TIMELINE_LIMIT",
    "MAX_RUNTIME_INSPECTION_TIMELINE_LIMIT",
    "RuntimeInspectionCompleteness",
    "RuntimeInspectionDiagnosticFinding",
    "RuntimeInspectionDiagnosticReadPort",
    "RuntimeInspectionDiagnosticSection",
    "RuntimeInspectionError",
    "RuntimeInspectionErrorCode",
    "RuntimeInspectionEvidenceReadPort",
    "RuntimeInspectionEvidenceReference",
    "RuntimeInspectionEvidenceSection",
    "RuntimeInspectionExecutionFactsReader",
    "RuntimeInspectionExecutionScope",
    "RuntimeInspectionExecutionScopeReader",
    "RuntimeInspectionExecutionStateSection",
    "RuntimeInspectionIdentitySection",
    "RuntimeInspectionNotFoundError",
    "RuntimeInspectionQuery",
    "RuntimeInspectionReadPort",
    "RuntimeInspectionScopeLookupOutcome",
    "RuntimeInspectionScopeLookupResult",
    "RuntimeInspectionSnapshot",
    "RuntimeInspectionSourceFailure",
    "RuntimeInspectionSourceFailureCode",
    "RuntimeInspectionTenantBoundaryError",
    "RuntimeInspectionTimelineDomain",
    "RuntimeInspectionTimelineEntry",
    "RuntimeInspectionTimelineSection",
]
