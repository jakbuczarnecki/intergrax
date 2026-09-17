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
from intergrax.contracts.runtime_inspection.limits import (
    DEFAULT_RUNTIME_INSPECTION_CONTINUATION_EPISODE_LIMIT,
    DEFAULT_RUNTIME_INSPECTION_GOVERNANCE_DECISION_LIMIT,
    DEFAULT_RUNTIME_INSPECTION_TOOL_INVOCATION_LIMIT,
)
from intergrax.contracts.runtime_inspection.sections import (
    RuntimeInspectionContinuationSection,
    RuntimeInspectionDiagnosticFinding,
    RuntimeInspectionDiagnosticSection,
    RuntimeInspectionEvidenceReference,
    RuntimeInspectionEvidenceSection,
    RuntimeInspectionExecutionStateSection,
    RuntimeInspectionGovernanceDecisionEntry,
    RuntimeInspectionGovernanceSection,
    RuntimeInspectionIdentitySection,
    RuntimeInspectionTimelineDomain,
    RuntimeInspectionTimelineEntry,
    RuntimeInspectionTimelineSection,
    RuntimeInspectionToolInvocation,
    RuntimeInspectionToolSection,
)
from intergrax.contracts.runtime_inspection.snapshot import RuntimeInspectionSnapshot
from intergrax.contracts.runtime_inspection.sources import (
    RuntimeInspectionContinuationReadPort,
    RuntimeInspectionDiagnosticReadPort,
    RuntimeInspectionEvidenceReadPort,
    RuntimeInspectionExecutionFactsReader,
    RuntimeInspectionExecutionScope,
    RuntimeInspectionExecutionScopeReader,
    RuntimeInspectionGovernanceReadPort,
    RuntimeInspectionScopeLookupOutcome,
    RuntimeInspectionScopeLookupResult,
    RuntimeInspectionToolReadPort,
)

__all__ = [
    "DEFAULT_RUNTIME_INSPECTION_CONTINUATION_EPISODE_LIMIT",
    "DEFAULT_RUNTIME_INSPECTION_GOVERNANCE_DECISION_LIMIT",
    "DEFAULT_RUNTIME_INSPECTION_TIMELINE_LIMIT",
    "DEFAULT_RUNTIME_INSPECTION_TOOL_INVOCATION_LIMIT",
    "MAX_RUNTIME_INSPECTION_TIMELINE_LIMIT",
    "RuntimeInspectionCompleteness",
    "RuntimeInspectionContinuationReadPort",
    "RuntimeInspectionContinuationSection",
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
    "RuntimeInspectionGovernanceDecisionEntry",
    "RuntimeInspectionGovernanceReadPort",
    "RuntimeInspectionGovernanceSection",
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
    "RuntimeInspectionToolInvocation",
    "RuntimeInspectionToolReadPort",
    "RuntimeInspectionToolSection",
    "RuntimeInspectionTimelineDomain",
    "RuntimeInspectionTimelineEntry",
    "RuntimeInspectionTimelineSection",
]
