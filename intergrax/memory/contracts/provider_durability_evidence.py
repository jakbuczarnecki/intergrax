# © Artur Czarnecki. All rights reserved.

"""Platform-owned memory provider durability evidence (MEM-FINAL-AUDIT-5A-R3)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from intergrax.memory.contracts.provider_qualification import MemoryProviderCapabilityKind

__all__ = [
    "MemoryProviderTrustedDurabilityStatus",
    "MemoryProviderDurabilityEvidence",
    "MemoryProviderDurabilityEvidenceLookup",
    "MemoryProviderDurabilityEvidenceRegistry",
    "MemoryProviderDurabilityEvidenceResolveStatus",
    "MemoryProviderDurabilityProofKind",
    "TrustedMemoryProviderDurabilityEvidence",
    "durability_evidence_from_reopen_proof",
]


class MemoryProviderTrustedDurabilityStatus(StrEnum):
    DURABLE = "durable"
    NOT_DURABLE = "not_durable"
    UNKNOWN = "unknown"


class MemoryProviderDurabilityEvidenceResolveStatus(StrEnum):
    FOUND = "found"
    MISSING = "missing"
    AMBIGUOUS = "ambiguous"
    MISMATCH = "mismatch"


class MemoryProviderDurabilityProofKind(StrEnum):
    RESTART_REOPEN = "restart_reopen"
    REFERENCE_DURABLE = "reference_durable"
    REAL_VENDOR_RESTART = "real_vendor_restart"


@dataclass(frozen=True, slots=True)
class MemoryProviderDurabilityEvidence:
    """Immutable platform-owned durability proof for one provider capability."""

    provider_id: str
    capability: MemoryProviderCapabilityKind
    durability_status: MemoryProviderTrustedDurabilityStatus
    qualification_run_id: str
    reference_time_iso: str
    evidence_source: str
    proof_kind: MemoryProviderDurabilityProofKind
    provider_version: str | None = None


@dataclass(frozen=True, slots=True)
class MemoryProviderDurabilityEvidenceLookup:
    resolve_status: MemoryProviderDurabilityEvidenceResolveStatus
    evidence: MemoryProviderDurabilityEvidence | None = None


class MemoryProviderDurabilityEvidenceRegistry(Protocol):
    """Replaceable trusted durability evidence source (no global mutable singleton)."""

    def resolve(
        self,
        provider_id: str,
        capability: MemoryProviderCapabilityKind,
        provider_version: str | None = None,
    ) -> MemoryProviderDurabilityEvidenceLookup: ...


class TrustedMemoryProviderDurabilityEvidence:
    """Adapters from offline durability qualification artifacts to runtime evidence."""

    @staticmethod
    def from_reopen_proof(
        *,
        provider_id: str,
        capability: MemoryProviderCapabilityKind,
        qualification_run_id: str,
        reference_time_iso: str,
        reopen_passed: bool,
        delete_reopen_passed: bool | None,
        provider_version: str | None = None,
        evidence_source: str = "user_profile_reopen_qualification",
        proof_kind: MemoryProviderDurabilityProofKind = (
            MemoryProviderDurabilityProofKind.RESTART_REOPEN
        ),
    ) -> MemoryProviderDurabilityEvidence:
        if reopen_passed and delete_reopen_passed is True:
            status = MemoryProviderTrustedDurabilityStatus.DURABLE
        elif reopen_passed is False or delete_reopen_passed is False:
            status = MemoryProviderTrustedDurabilityStatus.NOT_DURABLE
        else:
            status = MemoryProviderTrustedDurabilityStatus.UNKNOWN
        return MemoryProviderDurabilityEvidence(
            provider_id=provider_id,
            capability=capability,
            durability_status=status,
            qualification_run_id=qualification_run_id,
            reference_time_iso=reference_time_iso,
            evidence_source=evidence_source,
            proof_kind=proof_kind,
            provider_version=provider_version,
        )


def durability_evidence_from_reopen_proof(
    *,
    provider_id: str,
    capability: MemoryProviderCapabilityKind,
    qualification_run_id: str,
    reference_time_iso: str,
    reopen_passed: bool,
    delete_reopen_passed: bool | None,
    provider_version: str | None = None,
    evidence_source: str = "user_profile_reopen_qualification",
    proof_kind: MemoryProviderDurabilityProofKind = MemoryProviderDurabilityProofKind.RESTART_REOPEN,
) -> MemoryProviderDurabilityEvidence:
    return TrustedMemoryProviderDurabilityEvidence.from_reopen_proof(
        provider_id=provider_id,
        capability=capability,
        qualification_run_id=qualification_run_id,
        reference_time_iso=reference_time_iso,
        reopen_passed=reopen_passed,
        delete_reopen_passed=delete_reopen_passed,
        provider_version=provider_version,
        evidence_source=evidence_source,
        proof_kind=proof_kind,
    )
