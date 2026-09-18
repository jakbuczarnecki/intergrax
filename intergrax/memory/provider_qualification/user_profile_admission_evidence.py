# © Artur Czarnecki. All rights reserved.

"""Trusted behavioral + durability evidence bundle from durable qualification (MEM-FINAL-AUDIT-5B)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.memory.contracts.provider_admission_evidence import (
    MemoryProviderAdmissionEvidenceContext,
)
from intergrax.memory.contracts.provider_durability_evidence import (
    MemoryProviderDurabilityProofKind,
    durability_evidence_from_reopen_proof,
)
from intergrax.memory.contracts.provider_qualification import (
    MemoryProviderCapabilityKind,
    MemoryProviderQualificationResult,
)
from intergrax.memory.contracts.provider_qualification_evidence import (
    qualification_evidence_from_result,
)
from intergrax.memory.provider_qualification import (
    InMemoryMemoryProviderDurabilityEvidenceRegistry,
    InMemoryMemoryProviderQualificationEvidenceRegistry,
)

__all__ = [
    "UserProfileDurableQualificationAdmissionBundle",
    "build_user_profile_admission_evidence_from_durable_qualification",
]


@dataclass(frozen=True, slots=True)
class UserProfileDurableQualificationAdmissionBundle:
    """Atomically paired behavioral + durability evidence for one qualification run."""

    admission_evidence: MemoryProviderAdmissionEvidenceContext
    qualification_run_id: str
    production_durable_qualified: bool


def build_user_profile_admission_evidence_from_durable_qualification(
    *,
    canonical: MemoryProviderQualificationResult,
    reopen_passed: bool | None,
    delete_reopen_passed: bool | None,
    production_durable_qualified: bool,
    capability: MemoryProviderCapabilityKind = MemoryProviderCapabilityKind.USER_PROFILE_STORE,
    durability_evidence_source: str = "user_profile_durable_qualification",
    proof_kind: MemoryProviderDurabilityProofKind = (
        MemoryProviderDurabilityProofKind.RESTART_REOPEN
    ),
) -> UserProfileDurableQualificationAdmissionBundle:
    """Map offline durable harness output into registries consumable by production admission."""
    behavioral = qualification_evidence_from_result(canonical, capability=capability)
    if behavioral is None:
        raise ValueError("canonical qualification result did not produce behavioral evidence")

    durability = durability_evidence_from_reopen_proof(
        provider_id=canonical.descriptor.provider_id,
        capability=capability,
        qualification_run_id=canonical.qualification_run_id,
        reference_time_iso=canonical.reference_time_iso,
        reopen_passed=bool(reopen_passed),
        delete_reopen_passed=delete_reopen_passed,
        provider_version=canonical.descriptor.provider_version,
        evidence_source=durability_evidence_source,
        proof_kind=proof_kind,
    )

    qual_registry = InMemoryMemoryProviderQualificationEvidenceRegistry()
    qual_registry.register(behavioral)
    dur_registry = InMemoryMemoryProviderDurabilityEvidenceRegistry()
    dur_registry.register(durability)

    return UserProfileDurableQualificationAdmissionBundle(
        admission_evidence=MemoryProviderAdmissionEvidenceContext(
            qualification_registry=qual_registry,
            durability_registry=dur_registry,
        ),
        qualification_run_id=canonical.qualification_run_id,
        production_durable_qualified=production_durable_qualified,
    )
