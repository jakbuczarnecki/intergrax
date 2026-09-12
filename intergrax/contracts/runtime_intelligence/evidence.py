# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Traceable evidence pointers for runtime intelligence (W6-B)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class IntelligenceEvidenceSourceKind(StrEnum):
    """Stable source kinds for canonical fact stores — not a second evidence authority."""

    RUNTIME_EVENT = "runtime_event"
    CHECKPOINT = "checkpoint"
    TERMINAL = "terminal"
    LINEAGE = "lineage"
    DIAGNOSTIC_READ = "diagnostic_read"
    OBSERVABILITY_EXPORT = "observability_export"


@dataclass(frozen=True, slots=True)
class IntelligenceEvidence:
    """One auditable link from analysis output back to observed facts."""

    evidence_id: str
    source_kind: IntelligenceEvidenceSourceKind
    source_ref: str
    relation: str
    summary: str = ""

    def __post_init__(self) -> None:
        for name, value in (
            ("evidence_id", self.evidence_id),
            ("source_ref", self.source_ref),
            ("relation", self.relation),
        ):
            if not value.strip():
                raise ValueError(f"{name} must be non-empty")


__all__ = [
    "IntelligenceEvidence",
    "IntelligenceEvidenceSourceKind",
]
