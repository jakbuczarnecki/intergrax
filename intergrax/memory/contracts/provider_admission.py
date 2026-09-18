# © Artur Czarnecki. All rights reserved.

"""Production memory provider admission contracts (MEM-FINAL-AUDIT-5A)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, runtime_checkable

from intergrax.memory.contracts.provider_qualification import (
    MemoryProviderCapabilityKind,
    MemoryProviderQualificationStatus,
)

__all__ = [
    "MemoryProviderAdmissionReasonCode",
    "MemoryProviderDurability",
    "MemoryStoreProviderMetadata",
    "UserProfileStoreProviderClassification",
    "classify_user_profile_store_provider",
]


class MemoryProviderDurability(StrEnum):
    EPHEMERAL = "ephemeral"
    DURABLE = "durable"
    UNKNOWN = "unknown"


class MemoryProviderAdmissionReasonCode(StrEnum):
    PROVIDER_NOT_DURABLE = "provider_not_durable"
    PROVIDER_NOT_QUALIFIED = "provider_not_qualified"
    PROVIDER_MISSING = "provider_missing"
    REFERENCE_PROVIDER_NOT_ADMISSIBLE = "reference_provider_not_admissible"


@runtime_checkable
class MemoryStoreProviderMetadata(Protocol):
    """Typed durability and qualification evidence for a materialized store."""

    @property
    def memory_provider_id(self) -> str: ...

    @property
    def memory_provider_durability(self) -> MemoryProviderDurability: ...

    @property
    def memory_provider_reference_only(self) -> bool: ...

    @property
    def memory_provider_qualification_status(self) -> MemoryProviderQualificationStatus: ...


@dataclass(frozen=True, slots=True)
class UserProfileStoreProviderClassification:
    capability: MemoryProviderCapabilityKind
    provider_id: str
    durability: MemoryProviderDurability
    reference_only: bool
    qualification_status: MemoryProviderQualificationStatus


def classify_user_profile_store_provider(
    store: object,
) -> UserProfileStoreProviderClassification:
    """Classify a materialized ``UserProfileStore`` using public provider metadata."""
    if isinstance(store, MemoryStoreProviderMetadata):
        return UserProfileStoreProviderClassification(
            capability=MemoryProviderCapabilityKind.USER_PROFILE_STORE,
            provider_id=store.memory_provider_id,
            durability=store.memory_provider_durability,
            reference_only=store.memory_provider_reference_only,
            qualification_status=store.memory_provider_qualification_status,
        )
    return UserProfileStoreProviderClassification(
        capability=MemoryProviderCapabilityKind.USER_PROFILE_STORE,
        provider_id="unknown",
        durability=MemoryProviderDurability.UNKNOWN,
        reference_only=False,
        qualification_status=MemoryProviderQualificationStatus.NOT_QUALIFIED,
    )


class MemoryProviderAdmissionError(RuntimeError):
    """Fail-closed admission error for persistent production USER/LTM memory."""

    def __init__(
        self,
        *,
        capability: MemoryProviderCapabilityKind,
        execution_mode: str,
        application_profile: str,
        reason_code: MemoryProviderAdmissionReasonCode,
        provider_id: str,
        durability: MemoryProviderDurability,
        qualification_status: MemoryProviderQualificationStatus,
        reference_only: bool,
    ) -> None:
        self.capability = capability
        self.execution_mode = execution_mode
        self.application_profile = application_profile
        self.reason_code = reason_code
        self.provider_id = provider_id
        self.durability = durability
        self.qualification_status = qualification_status
        self.reference_only = reference_only
        super().__init__(
            "memory provider admission failed: "
            f"capability={capability.value}; "
            f"execution_mode={execution_mode}; "
            f"application_profile={application_profile}; "
            f"reason_code={reason_code.value}; "
            f"provider_id={provider_id}; "
            f"durability={durability.value}; "
            f"qualification_status={qualification_status.value}; "
            f"reference_only={reference_only}",
        )
