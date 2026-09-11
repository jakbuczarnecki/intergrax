# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Canonical operator investigation read contracts (DIAG R7)."""

from __future__ import annotations

from enum import StrEnum


class DiagnosticEvidenceExplanationConfidence(StrEnum):
    """Evidence support level for operator explanations — not root-cause authority."""

    PROVEN = "proven"
    SUPPORTED = "supported"
    UNKNOWN = "unknown"


class DiagnosticRootCauseStatus(StrEnum):
    """Root cause is never inferred heuristically by the read layer."""

    UNKNOWN = "unknown"
    NOT_APPLICABLE = "not_applicable"


class DiagnosticInvestigationSeverity(StrEnum):
    """Operator-facing severity projection from Problem + proven failure facts."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNKNOWN = "unknown"


class DiagnosticImpactNodeHealth(StrEnum):
    """Lineage-derived impact status — not a second execution graph store."""

    HEALTHY = "healthy"
    FAILED = "failed"
    SKIPPED = "skipped"
    UNKNOWN = "unknown"


class DiagnosticRecommendationKind(StrEnum):
    """Typed remediation hints — recommendations only, never automatic actions."""

    INVESTIGATE_EVIDENCE_GAP = "investigate_evidence_gap"
    REVIEW_EXTENSION_PLUGIN = "review_extension_plugin"
    REVIEW_DECISION_CONTEXT = "review_decision_context"
    REVIEW_EXTERNAL_DEPENDENCY = "review_external_dependency"
    REVIEW_EXECUTION_LINEAGE = "review_execution_lineage"


__all__ = [
    "DiagnosticEvidenceExplanationConfidence",
    "DiagnosticImpactNodeHealth",
    "DiagnosticInvestigationSeverity",
    "DiagnosticRecommendationKind",
    "DiagnosticRootCauseStatus",
]
