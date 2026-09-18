# © Artur Czarnecki. All rights reserved.

"""Platform-owned memory provider qualification evidence (MEM-FINAL-AUDIT-5A-R)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from intergrax.memory.contracts.provider_qualification import (
    MemoryProviderCapabilityKind,
    MemoryProviderQualificationResult,
    MemoryProviderQualificationStatus,
)

__all__ = [
    "MemoryProviderQualificationEvidence",
    "MemoryProviderQualificationEvidenceLookup",
    "MemoryProviderQualificationEvidenceRegistry",
    "MemoryProviderQualificationEvidenceResolveStatus",
    "TrustedMemoryProviderQualificationEvidence",
    "qualification_evidence_from_result",
]


class MemoryProviderQualificationEvidenceResolveStatus(StrEnum):
    FOUND = "found"
    MISSING = "missing"
    AMBIGUOUS = "ambiguous"
    MISMATCH = "mismatch"


@dataclass(frozen=True, slots=True)
class MemoryProviderQualificationEvidence:
    """Immutable platform-owned qualification proof for one provider capability."""

    provider_id: str
    capability: MemoryProviderCapabilityKind
    status: MemoryProviderQualificationStatus
    qualification_run_id: str
    reference_time_iso: str
    evidence_source: str
    provider_version: str | None = None


@dataclass(frozen=True, slots=True)
class MemoryProviderQualificationEvidenceLookup:
    """Result of a trusted evidence registry lookup."""

    resolve_status: MemoryProviderQualificationEvidenceResolveStatus
    evidence: MemoryProviderQualificationEvidence | None = None


class MemoryProviderQualificationEvidenceRegistry(Protocol):
    """Replaceable trusted qualification evidence source (no global mutable singleton)."""

    def resolve(
        self,
        provider_id: str,
        capability: MemoryProviderCapabilityKind,
        provider_version: str | None = None,
    ) -> MemoryProviderQualificationEvidenceLookup: ...


class TrustedMemoryProviderQualificationEvidence:
    """Adapters from offline qualification artifacts to runtime evidence records."""

    @staticmethod
    def from_qualification_result(
        result: MemoryProviderQualificationResult,
        *,
        capability: MemoryProviderCapabilityKind,
        evidence_source: str = "memory_provider_qualification_runner",
    ) -> MemoryProviderQualificationEvidence | None:
        capability_result = next(
            (item for item in result.capability_results if item.capability is capability),
            None,
        )
        if capability_result is None:
            return None
        return MemoryProviderQualificationEvidence(
            provider_id=result.descriptor.provider_id,
            capability=capability,
            status=capability_result.status,
            qualification_run_id=result.qualification_run_id,
            reference_time_iso=result.reference_time_iso,
            evidence_source=evidence_source,
            provider_version=result.descriptor.provider_version,
        )


def qualification_evidence_from_result(
    result: MemoryProviderQualificationResult,
    *,
    capability: MemoryProviderCapabilityKind,
    evidence_source: str = "memory_provider_qualification_runner",
) -> MemoryProviderQualificationEvidence | None:
    return TrustedMemoryProviderQualificationEvidence.from_qualification_result(
        result,
        capability=capability,
        evidence_source=evidence_source,
    )
