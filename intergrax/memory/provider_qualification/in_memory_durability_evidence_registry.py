# © Artur Czarnecki. All rights reserved.

"""In-memory durability evidence registry (tests and composition reference)."""

from __future__ import annotations

from intergrax.memory.contracts.provider_durability_evidence import (
    MemoryProviderDurabilityEvidence,
    MemoryProviderDurabilityEvidenceLookup,
    MemoryProviderDurabilityEvidenceResolveStatus,
)
from intergrax.memory.contracts.provider_qualification import MemoryProviderCapabilityKind

__all__ = ["InMemoryMemoryProviderDurabilityEvidenceRegistry"]


class InMemoryMemoryProviderDurabilityEvidenceRegistry:
    """Platform-composed registry; providers cannot self-register."""

    def __init__(self, records: tuple[MemoryProviderDurabilityEvidence, ...] = ()) -> None:
        self._records = tuple(records)

    def register(self, evidence: MemoryProviderDurabilityEvidence) -> None:
        self._records = self._records + (evidence,)

    def resolve(
        self,
        provider_id: str,
        capability: MemoryProviderCapabilityKind,
        provider_version: str | None = None,
    ) -> MemoryProviderDurabilityEvidenceLookup:
        matches = [
            item
            for item in self._records
            if item.provider_id == provider_id and item.capability is capability
        ]
        if not matches:
            return MemoryProviderDurabilityEvidenceLookup(
                resolve_status=MemoryProviderDurabilityEvidenceResolveStatus.MISSING,
            )
        if provider_version is not None:
            version_matches = [
                item
                for item in matches
                if item.provider_version == provider_version
            ]
            if not version_matches:
                return MemoryProviderDurabilityEvidenceLookup(
                    resolve_status=MemoryProviderDurabilityEvidenceResolveStatus.MISMATCH,
                )
            matches = version_matches
        if len(matches) > 1:
            statuses = {item.durability_status for item in matches}
            run_ids = {item.qualification_run_id for item in matches}
            if len(statuses) > 1 or len(run_ids) > 1:
                return MemoryProviderDurabilityEvidenceLookup(
                    resolve_status=MemoryProviderDurabilityEvidenceResolveStatus.AMBIGUOUS,
                )
        return MemoryProviderDurabilityEvidenceLookup(
            resolve_status=MemoryProviderDurabilityEvidenceResolveStatus.FOUND,
            evidence=matches[0],
        )
