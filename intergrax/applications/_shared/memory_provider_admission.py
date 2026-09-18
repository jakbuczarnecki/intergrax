# © Artur Czarnecki. All rights reserved.

"""Host-owned memory provider admission (MEM-FINAL-AUDIT-5A)."""

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
)
from intergrax.memory.contracts.provider_qualification import MemoryProviderQualificationStatus
from intergrax.memory.user_profile_store import UserProfileStore


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
    """Replaceable admission strategy; hard invariant enforced by default product policy."""

    def evaluate_user_profile_store(
        self,
        classification: UserProfileStoreProviderClassification,
    ) -> MemoryProviderAdmissionDecision: ...


@dataclass(frozen=True, slots=True)
class DefaultProductionMemoryProviderAdmissionPolicy:
    """Production persistent USER/LTM requires durable, non-reference, qualified provider."""

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
        if classification.reference_only:
            return MemoryProviderAdmissionDecision(
                admitted=False,
                reason_code=MemoryProviderAdmissionReasonCode.REFERENCE_PROVIDER_NOT_ADMISSIBLE,
            )
        if classification.durability is not MemoryProviderDurability.DURABLE:
            return MemoryProviderAdmissionDecision(
                admitted=False,
                reason_code=MemoryProviderAdmissionReasonCode.PROVIDER_NOT_DURABLE,
            )
        if classification.qualification_status is not MemoryProviderQualificationStatus.QUALIFIED:
            return MemoryProviderAdmissionDecision(
                admitted=False,
                reason_code=MemoryProviderAdmissionReasonCode.PROVIDER_NOT_QUALIFIED,
            )
        return MemoryProviderAdmissionDecision(admitted=True, reason_code=None)


def validate_memory_platform_wiring_admission(
    env: ApplicationEnvironmentProfile,
    user_profile_store: UserProfileStore,
    *,
    policy: MemoryProviderAdmissionPolicy | None = None,
) -> None:
    """Pure admission gate on final user profile store; no provider mutation."""
    if not persistent_canonical_user_profile_memory_required(env):
        return
    resolved_policy = policy or DefaultProductionMemoryProviderAdmissionPolicy.for_environment(
        env,
    )
    classification = classify_user_profile_store_provider(user_profile_store)
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
        qualification_status=classification.qualification_status,
        reference_only=classification.reference_only,
    )
