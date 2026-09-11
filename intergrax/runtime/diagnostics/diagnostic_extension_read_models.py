# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Operator read projections for diagnostic extensions (R5)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from intergrax.contracts.diagnostic_analyzer import DiagnosticExtensionCertainty
from intergrax.contracts.execution_identity import EventId


class DiagnosticExtensionReadStatus(StrEnum):
    """Whether extension enrichment completed for one occurrence."""

    COMPLETE = "complete"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"


class DiagnosticExtensionPluginStatus(StrEnum):
    """Per-plugin outcome — faulty plugins cannot fail the central engine."""

    AVAILABLE = "available"
    PLUGIN_UNAVAILABLE = "plugin_unavailable"


@dataclass(frozen=True, slots=True)
class DiagnosticExtensionEvidenceView:
    evidence_id: EventId
    evidence_namespace: str
    kind: str
    summary: str


@dataclass(frozen=True, slots=True)
class DiagnosticExtensionFindingView:
    """Additional domain interpretation — never a Problem."""

    analyzer_id: str
    analyzer_namespace: str
    kind: str
    certainty: DiagnosticExtensionCertainty
    summary: str
    evidence_refs: tuple[EventId, ...]
    plugin_status: DiagnosticExtensionPluginStatus = (
        DiagnosticExtensionPluginStatus.AVAILABLE
    )


@dataclass(frozen=True, slots=True)
class DiagnosticExtensionOccurrenceEnrichment:
    read_status: DiagnosticExtensionReadStatus
    contributed_evidence: tuple[DiagnosticExtensionEvidenceView, ...]
    extension_findings: tuple[DiagnosticExtensionFindingView, ...]
    limitations: tuple[str, ...] = ()


__all__ = [
    "DiagnosticExtensionEvidenceView",
    "DiagnosticExtensionFindingView",
    "DiagnosticExtensionOccurrenceEnrichment",
    "DiagnosticExtensionPluginStatus",
    "DiagnosticExtensionReadStatus",
]
