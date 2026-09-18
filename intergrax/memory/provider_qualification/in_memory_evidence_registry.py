# © Artur Czarnecki. All rights reserved.

"""In-memory qualification evidence registry (tests and composition reference)."""

from __future__ import annotations

from intergrax.memory.contracts.provider_qualification import MemoryProviderCapabilityKind
from intergrax.memory.contracts.provider_qualification_evidence import (
    MemoryProviderQualificationEvidence,
    MemoryProviderQualificationEvidenceLookup,
    MemoryProviderQualificationEvidenceResolveStatus,
)

__all__ = ["InMemoryMemoryProviderQualificationEvidenceRegistry"]


class InMemoryMemoryProviderQualificationEvidenceRegistry:
    """Platform-composed registry; providers cannot self-register."""

    def __init__(self, records: tuple[MemoryProviderQualificationEvidence, ...] = ()) -> None:
        self._records = tuple(records)

    def register(self, evidence: MemoryProviderQualificationEvidence) -> None:
        self._records = self._records + (evidence,)

    def resolve(
        self,
        provider_id: str,
        capability: MemoryProviderCapabilityKind,
        provider_version: str | None = None,
        backing_provider_id: str | None = None,
        backing_provider_version: str | None = None,
    ) -> MemoryProviderQualificationEvidenceLookup:
        matches = [
            item
            for item in self._records
            if item.provider_id == provider_id
            and item.capability is capability
            and item.backing_provider_id == backing_provider_id
            and item.backing_provider_version == backing_provider_version
        ]
        if not matches:
            return MemoryProviderQualificationEvidenceLookup(
                resolve_status=MemoryProviderQualificationEvidenceResolveStatus.MISSING,
            )
        if provider_version is not None:
            version_matches = [
                item
                for item in matches
                if item.provider_version == provider_version
            ]
            if not version_matches:
                return MemoryProviderQualificationEvidenceLookup(
                    resolve_status=MemoryProviderQualificationEvidenceResolveStatus.MISMATCH,
                )
            matches = version_matches
        if len(matches) > 1:
            statuses = {item.status for item in matches}
            run_ids = {item.qualification_run_id for item in matches}
            if len(statuses) > 1 or len(run_ids) > 1:
                return MemoryProviderQualificationEvidenceLookup(
                    resolve_status=MemoryProviderQualificationEvidenceResolveStatus.AMBIGUOUS,
                )
        return MemoryProviderQualificationEvidenceLookup(
            resolve_status=MemoryProviderQualificationEvidenceResolveStatus.FOUND,
            evidence=matches[0],
        )
