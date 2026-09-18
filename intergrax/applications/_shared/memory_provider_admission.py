# © Artur Czarnecki. All rights reserved.

"""Host-owned memory provider admission (MEM-FINAL-AUDIT-5A / 5A-R)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from intergrax.applications.contracts.application_host import ApplicationProfile
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.memory.contracts.provider_admission import (
    MemoryProviderAdmissionError,
    MemoryProviderAdmissionReasonCode,
    MemoryProviderDurability,
    UserProfileStoreProviderClassification,
    classify_user_profile_store_provider,
    evaluate_production_persistent_user_profile_admission,
    lookup_trusted_user_profile_qualification_evidence,
)
from intergrax.memory.contracts.provider_identity import MemoryProviderIdentity
from intergrax.memory.contracts.provider_qualification_evidence import (
    MemoryProviderQualificationEvidenceLookup,
    MemoryProviderQualificationEvidenceRegistry,
    MemoryProviderQualificationEvidenceResolveStatus,
)
from intergrax.memory.user_profile_store import UserProfileStore


class _EmptyQualificationEvidenceRegistry:
    def resolve(self, provider_id: str, capability: object, provider_version: str | None = None):
        from intergrax.memory.contracts.provider_qualification_evidence import (
            MemoryProviderQualificationEvidenceLookup,
            MemoryProviderQualificationEvidenceResolveStatus,
        )

        _ = (provider_id, capability, provider_version)
        return MemoryProviderQualificationEvidenceLookup(
            resolve_status=MemoryProviderQualificationEvidenceResolveStatus.MISSING,
        )


def persistent_canonical_user_profile_memory_required(
    env: ApplicationEnvironmentProfile,
) -> bool:
    memory_profile = env.memory_profile
    return (
        memory_profile.enable_user_memory or memory_profile.enable_long_term_memory
    )


@dataclass(frozen=True, slots=True)
class MemoryProviderAdmissionDecision:
    admitted: bool
    reason_code: MemoryProviderAdmissionReasonCode | None


class MemoryProviderAdmissionPolicy(Protocol):
    """Replaceable admission strategy for non-product paths (LAB / harness)."""

    def evaluate_user_profile_store(
        self,
        classification: UserProfileStoreProviderClassification,
    ) -> MemoryProviderAdmissionDecision: ...


@dataclass(frozen=True, slots=True)
class DefaultProductionMemoryProviderAdmissionPolicy:
    """LAB harness: admission not enforced when ``admission_enforced`` is false."""

    admission_enforced: bool

    @classmethod
    def for_environment(
        cls,
        env: ApplicationEnvironmentProfile,
    ) -> DefaultProductionMemoryProviderAdmissionPolicy:
        return cls(
            admission_enforced=env.application_profile is ApplicationProfile.PRODUCT,
        )

    def evaluate_user_profile_store(
        self,
        classification: UserProfileStoreProviderClassification,
    ) -> MemoryProviderAdmissionDecision:
        if not self.admission_enforced:
            return MemoryProviderAdmissionDecision(admitted=True, reason_code=None)
        missing = MemoryProviderQualificationEvidenceLookup(
            resolve_status=MemoryProviderQualificationEvidenceResolveStatus.MISSING,
        )
        evaluation = evaluate_production_persistent_user_profile_admission(
            classification,
            None,
            missing,
        )
        return MemoryProviderAdmissionDecision(
            admitted=evaluation.admitted,
            reason_code=evaluation.reason_code,
        )


def validate_memory_platform_wiring_admission(
    env: ApplicationEnvironmentProfile,
    user_profile_store: UserProfileStore,
    *,
    user_profile_store_identity: MemoryProviderIdentity | None = None,
    policy: MemoryProviderAdmissionPolicy | None = None,
    qualification_evidence_registry: MemoryProviderQualificationEvidenceRegistry | None = None,
) -> None:
    """Pure admission gate on final user profile store; no provider mutation."""
    if not persistent_canonical_user_profile_memory_required(env):
        return

    classification = classify_user_profile_store_provider(user_profile_store)
    registry = qualification_evidence_registry or _EmptyQualificationEvidenceRegistry()

    if env.application_profile is ApplicationProfile.PRODUCT:
        if user_profile_store_identity is None:
            evidence_lookup = MemoryProviderQualificationEvidenceLookup(
                resolve_status=MemoryProviderQualificationEvidenceResolveStatus.MISSING,
            )
        else:
            evidence_lookup = lookup_trusted_user_profile_qualification_evidence(
                registry,
                user_profile_store_identity,
            )
        evaluation = evaluate_production_persistent_user_profile_admission(
            classification,
            user_profile_store_identity,
            evidence_lookup,
        )
        if not evaluation.admitted:
            reason = evaluation.reason_code or MemoryProviderAdmissionReasonCode.PROVIDER_MISSING
            declared_id = classification.provider_id
            trusted_id = (
                user_profile_store_identity.provider_id
                if user_profile_store_identity is not None
                else None
            )
            raise MemoryProviderAdmissionError(
                capability=classification.capability,
                execution_mode=env.execution_mode.value,
                application_profile=env.application_profile.value,
                reason_code=reason,
                provider_id=evaluation.provider_id,
                durability=classification.durability,
                declared_qualification_status=classification.declared_qualification_status,
                trusted_qualification_status=evaluation.trusted_qualification_status,
                reference_only=classification.reference_only,
                qualification_run_id=evaluation.qualification_run_id,
                declared_provider_id=declared_id,
                trusted_provider_id=trusted_id,
            )
        return

    resolved_policy = policy or DefaultProductionMemoryProviderAdmissionPolicy.for_environment(
        env,
    )
    decision = resolved_policy.evaluate_user_profile_store(classification)
    if decision.admitted:
        return
    reason = decision.reason_code or MemoryProviderAdmissionReasonCode.PROVIDER_MISSING
    raise MemoryProviderAdmissionError(
        capability=classification.capability,
        execution_mode=env.execution_mode.value,
        application_profile=env.application_profile.value,
        reason_code=reason,
        provider_id=classification.provider_id,
        durability=classification.durability,
        declared_qualification_status=classification.declared_qualification_status,
        trusted_qualification_status=None,
        reference_only=classification.reference_only,
    )
