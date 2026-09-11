# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Domain diagnostic analyzer SPI — bounded finding candidates only (R5)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, runtime_checkable

from intergrax.contracts.diagnostic_extension_evidence import (
    DiagnosticExtensionEvidence,
)
from intergrax.contracts.execution_identity import EventId


class DiagnosticExtensionCertainty(StrEnum):
    """Operator-facing support level for extension findings — not global Problem certainty."""

    SUPPORTED = "supported"
    INCONCLUSIVE = "inconclusive"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True, slots=True)
class DiagnosticFindingScope:
    """Where an extension finding applies — never mints execution identity."""

    tenant_id: str
    evidence_namespace: str


@dataclass(frozen=True, slots=True)
class DiagnosticFindingCandidate:
    """Analyzer output — central engine projects to read model; never a Problem."""

    kind: str
    confidence: DiagnosticExtensionCertainty
    evidence_refs: tuple[EventId, ...]
    scope: DiagnosticFindingScope
    summary: str


@runtime_checkable
class DiagnosticAnalyzer(Protocol):
    """Map contributed evidence to bounded finding candidates."""

    @property
    def analyzer_id(self) -> str:
        """Stable analyzer identity for registry ordering and audit."""

    @property
    def analyzer_namespace(self) -> str:
        """Owning namespace (e.g. company.sap)."""

    @property
    def priority(self) -> int:
        """Lower values run earlier; ties broken by namespace then analyzer_id."""

    def analyze(
        self,
        evidence: tuple[DiagnosticExtensionEvidence, ...],
    ) -> tuple[DiagnosticFindingCandidate, ...]:
        """Pure analysis over in-scope evidence — no persistence or Problem writes."""


__all__ = [
    "DiagnosticAnalyzer",
    "DiagnosticExtensionCertainty",
    "DiagnosticFindingCandidate",
    "DiagnosticFindingScope",
]
