# © Artur Czarnecki. All rights reserved.

"""Production memory provider admission contracts (MEM-FINAL-AUDIT-5A / 5A-R)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, runtime_checkable

from intergrax.memory.contracts.provider_qualification import (
    MemoryProviderCapabilityKind,
    MemoryProviderQualificationStatus,
)
from intergrax.memory.contracts.provider_identity import (
    MemoryProviderIdentity,
    memory_provider_backing_identity_mismatch,
)
from intergrax.memory.contracts.provider_durability_evidence import (
    MemoryProviderDurabilityEvidence,
    MemoryProviderDurabilityEvidenceLookup,
    MemoryProviderDurabilityEvidenceRegistry,
    MemoryProviderDurabilityEvidenceResolveStatus,
    MemoryProviderTrustedDurabilityStatus,
)
from intergrax.memory.contracts.provider_qualification_evidence import (
    MemoryProviderQualificationEvidence,
    MemoryProviderQualificationEvidenceLookup,
    MemoryProviderQualificationEvidenceResolveStatus,
    MemoryProviderQualificationEvidenceRegistry,
)

__all__ = [
    "MemoryProviderAdmissionReasonCode",
    "MemoryProviderDurability",
    "MemoryStoreProviderMetadata",
    "UserProfileStoreProviderClassification",
    "UserProfileStoreProductionAdmissionEvaluation",
    "classify_user_profile_store_provider",
    "evaluate_production_persistent_user_profile_admission",
    "lookup_trusted_user_profile_durability_evidence",
    "lookup_trusted_user_profile_qualification_evidence",
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
    QUALIFICATION_EVIDENCE_MISSING = "qualification_evidence_missing"
    QUALIFICATION_EVIDENCE_MISMATCH = "qualification_evidence_mismatch"
    DURABILITY_EVIDENCE_MISSING = "durability_evidence_missing"
    DURABILITY_EVIDENCE_MISMATCH = "durability_evidence_mismatch"
    PROVIDER_IDENTITY_MISSING = "provider_identity_missing"
    PROVIDER_IDENTITY_MISMATCH = "provider_identity_mismatch"
    PROVIDER_BACKING_IDENTITY_MISMATCH = "provider_backing_identity_mismatch"


@runtime_checkable
class MemoryStoreProviderMetadata(Protocol):
    """Provider-declared metadata for a materialized store (not trusted qualification proof)."""

    @property
    def memory_provider_id(self) -> str: ...

    @property
    def memory_provider_durability(self) -> MemoryProviderDurability: ...

    @property
    def memory_provider_reference_only(self) -> bool: ...

    @property
    def memory_provider_declared_qualification_status(self) -> MemoryProviderQualificationStatus: ...

    @property
    def memory_provider_version(self) -> str | None:
        """Optional provider release/version when declared by the plugin."""
        ...


@dataclass(frozen=True, slots=True)
class UserProfileStoreProviderClassification:
    capability: MemoryProviderCapabilityKind
    provider_id: str
    durability: MemoryProviderDurability  # provider-declared claim; not trusted for admission
    reference_only: bool
    declared_qualification_status: MemoryProviderQualificationStatus
    provider_version: str | None = None


@dataclass(frozen=True, slots=True)
class UserProfileStoreProductionAdmissionEvaluation:
    admitted: bool
    reason_code: MemoryProviderAdmissionReasonCode | None
    provider_id: str
    declared_qualification_status: MemoryProviderQualificationStatus
    trusted_qualification_status: MemoryProviderQualificationStatus | None
    qualification_run_id: str | None
    trusted_durability_status: MemoryProviderTrustedDurabilityStatus | None = None
    durability_run_id: str | None = None


def classify_user_profile_store_provider(
    store: object,
) -> UserProfileStoreProviderClassification:
    """Classify a materialized ``UserProfileStore`` using public provider-declared metadata."""
    if isinstance(store, MemoryStoreProviderMetadata):
        return UserProfileStoreProviderClassification(
            capability=MemoryProviderCapabilityKind.USER_PROFILE_STORE,
            provider_id=store.memory_provider_id,
            durability=store.memory_provider_durability,
            reference_only=store.memory_provider_reference_only,
            declared_qualification_status=store.memory_provider_declared_qualification_status,
            provider_version=store.memory_provider_version,
        )
    return UserProfileStoreProviderClassification(
        capability=MemoryProviderCapabilityKind.USER_PROFILE_STORE,
        provider_id="unknown",
        durability=MemoryProviderDurability.UNKNOWN,
        reference_only=False,
        declared_qualification_status=MemoryProviderQualificationStatus.NOT_QUALIFIED,
    )


def lookup_trusted_user_profile_qualification_evidence(
    registry: MemoryProviderQualificationEvidenceRegistry,
    trusted_identity: MemoryProviderIdentity,
) -> MemoryProviderQualificationEvidenceLookup:
    return registry.resolve(
        trusted_identity.provider_id,
        trusted_identity.capability,
        trusted_identity.provider_version,
        trusted_identity.backing_provider_id,
        trusted_identity.backing_provider_version,
    )


def lookup_trusted_user_profile_durability_evidence(
    registry: MemoryProviderDurabilityEvidenceRegistry,
    trusted_identity: MemoryProviderIdentity,
) -> MemoryProviderDurabilityEvidenceLookup:
    return registry.resolve(
        trusted_identity.provider_id,
        trusted_identity.capability,
        trusted_identity.provider_version,
        trusted_identity.backing_provider_id,
        trusted_identity.backing_provider_version,
    )


def _provider_version_binding_mismatch_qualification(
    trusted_identity: MemoryProviderIdentity,
    evidence: MemoryProviderQualificationEvidence,
) -> bool:
    identity_version = trusted_identity.provider_version
    evidence_version = evidence.provider_version
    if identity_version is None and evidence_version is None:
        return False
    if identity_version is None or evidence_version is None:
        return True
    return identity_version != evidence_version


def _provider_version_binding_mismatch_durability(
    trusted_identity: MemoryProviderIdentity,
    evidence: MemoryProviderDurabilityEvidence,
) -> bool:
    identity_version = trusted_identity.provider_version
    evidence_version = evidence.provider_version
    if identity_version is None and evidence_version is None:
        return False
    if identity_version is None or evidence_version is None:
        return True
    return identity_version != evidence_version


def _provider_backing_binding_mismatch_qualification(
    trusted_identity: MemoryProviderIdentity,
    evidence: MemoryProviderQualificationEvidence,
) -> bool:
    return memory_provider_backing_identity_mismatch(
        trusted_identity,
        backing_provider_id=evidence.backing_provider_id,
        backing_provider_version=evidence.backing_provider_version,
    )


def _provider_backing_binding_mismatch_durability(
    trusted_identity: MemoryProviderIdentity,
    evidence: MemoryProviderDurabilityEvidence,
) -> bool:
    return memory_provider_backing_identity_mismatch(
        trusted_identity,
        backing_provider_id=evidence.backing_provider_id,
        backing_provider_version=evidence.backing_provider_version,
    )


def evaluate_production_persistent_user_profile_admission(
    classification: UserProfileStoreProviderClassification,
    trusted_identity: MemoryProviderIdentity | None,
    evidence_lookup: MemoryProviderQualificationEvidenceLookup,
    durability_lookup: MemoryProviderDurabilityEvidenceLookup,
) -> UserProfileStoreProductionAdmissionEvaluation:
    """Mandatory PRODUCT persistent gate; ignores self-declared qualification and durability."""
    trusted_provider_id = (
        trusted_identity.provider_id if trusted_identity is not None else "none"
    )
    if trusted_identity is None:
        return UserProfileStoreProductionAdmissionEvaluation(
            admitted=False,
            reason_code=MemoryProviderAdmissionReasonCode.PROVIDER_IDENTITY_MISSING,
            provider_id=classification.provider_id,
            declared_qualification_status=classification.declared_qualification_status,
            trusted_qualification_status=None,
            qualification_run_id=None,
        )
    if trusted_identity.capability is not MemoryProviderCapabilityKind.USER_PROFILE_STORE:
        return UserProfileStoreProductionAdmissionEvaluation(
            admitted=False,
            reason_code=MemoryProviderAdmissionReasonCode.PROVIDER_IDENTITY_MISMATCH,
            provider_id=trusted_provider_id,
            declared_qualification_status=classification.declared_qualification_status,
            trusted_qualification_status=None,
            qualification_run_id=None,
        )
    if classification.reference_only:
        return UserProfileStoreProductionAdmissionEvaluation(
            admitted=False,
            reason_code=MemoryProviderAdmissionReasonCode.REFERENCE_PROVIDER_NOT_ADMISSIBLE,
            provider_id=trusted_provider_id,
            declared_qualification_status=classification.declared_qualification_status,
            trusted_qualification_status=None,
            qualification_run_id=None,
        )
    if (
        classification.provider_id != "unknown"
        and classification.provider_id != trusted_identity.provider_id
    ):
        return UserProfileStoreProductionAdmissionEvaluation(
            admitted=False,
            reason_code=MemoryProviderAdmissionReasonCode.PROVIDER_IDENTITY_MISMATCH,
            provider_id=trusted_provider_id,
            declared_qualification_status=classification.declared_qualification_status,
            trusted_qualification_status=None,
            qualification_run_id=None,
        )
    if evidence_lookup.resolve_status is MemoryProviderQualificationEvidenceResolveStatus.MISSING:
        return UserProfileStoreProductionAdmissionEvaluation(
            admitted=False,
            reason_code=MemoryProviderAdmissionReasonCode.QUALIFICATION_EVIDENCE_MISSING,
            provider_id=classification.provider_id,
            declared_qualification_status=classification.declared_qualification_status,
            trusted_qualification_status=None,
            qualification_run_id=None,
        )
    if evidence_lookup.resolve_status in {
        MemoryProviderQualificationEvidenceResolveStatus.AMBIGUOUS,
        MemoryProviderQualificationEvidenceResolveStatus.MISMATCH,
    }:
        return UserProfileStoreProductionAdmissionEvaluation(
            admitted=False,
            reason_code=MemoryProviderAdmissionReasonCode.QUALIFICATION_EVIDENCE_MISMATCH,
            provider_id=classification.provider_id,
            declared_qualification_status=classification.declared_qualification_status,
            trusted_qualification_status=None,
            qualification_run_id=None,
        )
    evidence = evidence_lookup.evidence
    if evidence is None:
        return UserProfileStoreProductionAdmissionEvaluation(
            admitted=False,
            reason_code=MemoryProviderAdmissionReasonCode.QUALIFICATION_EVIDENCE_MISSING,
            provider_id=classification.provider_id,
            declared_qualification_status=classification.declared_qualification_status,
            trusted_qualification_status=None,
            qualification_run_id=None,
        )
    if evidence.capability is not classification.capability:
        return UserProfileStoreProductionAdmissionEvaluation(
            admitted=False,
            reason_code=MemoryProviderAdmissionReasonCode.QUALIFICATION_EVIDENCE_MISMATCH,
            provider_id=classification.provider_id,
            declared_qualification_status=classification.declared_qualification_status,
            trusted_qualification_status=evidence.status,
            qualification_run_id=evidence.qualification_run_id,
        )
    if evidence.provider_id != trusted_identity.provider_id:
        return UserProfileStoreProductionAdmissionEvaluation(
            admitted=False,
            reason_code=MemoryProviderAdmissionReasonCode.QUALIFICATION_EVIDENCE_MISMATCH,
            provider_id=trusted_provider_id,
            declared_qualification_status=classification.declared_qualification_status,
            trusted_qualification_status=evidence.status,
            qualification_run_id=evidence.qualification_run_id,
        )
    if _provider_backing_binding_mismatch_qualification(trusted_identity, evidence):
        return UserProfileStoreProductionAdmissionEvaluation(
            admitted=False,
            reason_code=MemoryProviderAdmissionReasonCode.PROVIDER_BACKING_IDENTITY_MISMATCH,
            provider_id=trusted_provider_id,
            declared_qualification_status=classification.declared_qualification_status,
            trusted_qualification_status=evidence.status,
            qualification_run_id=evidence.qualification_run_id,
        )
    if _provider_version_binding_mismatch_qualification(trusted_identity, evidence):
        return UserProfileStoreProductionAdmissionEvaluation(
            admitted=False,
            reason_code=MemoryProviderAdmissionReasonCode.QUALIFICATION_EVIDENCE_MISMATCH,
            provider_id=trusted_provider_id,
            declared_qualification_status=classification.declared_qualification_status,
            trusted_qualification_status=evidence.status,
            qualification_run_id=evidence.qualification_run_id,
        )
    if evidence.status is not MemoryProviderQualificationStatus.QUALIFIED:
        return UserProfileStoreProductionAdmissionEvaluation(
            admitted=False,
            reason_code=MemoryProviderAdmissionReasonCode.PROVIDER_NOT_QUALIFIED,
            provider_id=classification.provider_id,
            declared_qualification_status=classification.declared_qualification_status,
            trusted_qualification_status=evidence.status,
            qualification_run_id=evidence.qualification_run_id,
        )
    if durability_lookup.resolve_status is MemoryProviderDurabilityEvidenceResolveStatus.MISSING:
        return UserProfileStoreProductionAdmissionEvaluation(
            admitted=False,
            reason_code=MemoryProviderAdmissionReasonCode.DURABILITY_EVIDENCE_MISSING,
            provider_id=trusted_provider_id,
            declared_qualification_status=classification.declared_qualification_status,
            trusted_qualification_status=evidence.status,
            qualification_run_id=evidence.qualification_run_id,
            trusted_durability_status=None,
            durability_run_id=None,
        )
    if durability_lookup.resolve_status in {
        MemoryProviderDurabilityEvidenceResolveStatus.AMBIGUOUS,
        MemoryProviderDurabilityEvidenceResolveStatus.MISMATCH,
    }:
        return UserProfileStoreProductionAdmissionEvaluation(
            admitted=False,
            reason_code=MemoryProviderAdmissionReasonCode.DURABILITY_EVIDENCE_MISMATCH,
            provider_id=trusted_provider_id,
            declared_qualification_status=classification.declared_qualification_status,
            trusted_qualification_status=evidence.status,
            qualification_run_id=evidence.qualification_run_id,
            trusted_durability_status=None,
            durability_run_id=None,
        )
    durability_evidence = durability_lookup.evidence
    if durability_evidence is None:
        return UserProfileStoreProductionAdmissionEvaluation(
            admitted=False,
            reason_code=MemoryProviderAdmissionReasonCode.DURABILITY_EVIDENCE_MISSING,
            provider_id=trusted_provider_id,
            declared_qualification_status=classification.declared_qualification_status,
            trusted_qualification_status=evidence.status,
            qualification_run_id=evidence.qualification_run_id,
            trusted_durability_status=None,
            durability_run_id=None,
        )
    if durability_evidence.capability is not classification.capability:
        return UserProfileStoreProductionAdmissionEvaluation(
            admitted=False,
            reason_code=MemoryProviderAdmissionReasonCode.DURABILITY_EVIDENCE_MISMATCH,
            provider_id=trusted_provider_id,
            declared_qualification_status=classification.declared_qualification_status,
            trusted_qualification_status=evidence.status,
            qualification_run_id=evidence.qualification_run_id,
            trusted_durability_status=durability_evidence.durability_status,
            durability_run_id=durability_evidence.qualification_run_id,
        )
    if durability_evidence.provider_id != trusted_identity.provider_id:
        return UserProfileStoreProductionAdmissionEvaluation(
            admitted=False,
            reason_code=MemoryProviderAdmissionReasonCode.DURABILITY_EVIDENCE_MISMATCH,
            provider_id=trusted_provider_id,
            declared_qualification_status=classification.declared_qualification_status,
            trusted_qualification_status=evidence.status,
            qualification_run_id=evidence.qualification_run_id,
            trusted_durability_status=durability_evidence.durability_status,
            durability_run_id=durability_evidence.qualification_run_id,
        )
    if _provider_backing_binding_mismatch_durability(trusted_identity, durability_evidence):
        return UserProfileStoreProductionAdmissionEvaluation(
            admitted=False,
            reason_code=MemoryProviderAdmissionReasonCode.PROVIDER_BACKING_IDENTITY_MISMATCH,
            provider_id=trusted_provider_id,
            declared_qualification_status=classification.declared_qualification_status,
            trusted_qualification_status=evidence.status,
            qualification_run_id=evidence.qualification_run_id,
            trusted_durability_status=durability_evidence.durability_status,
            durability_run_id=durability_evidence.qualification_run_id,
        )
    if _provider_version_binding_mismatch_durability(trusted_identity, durability_evidence):
        return UserProfileStoreProductionAdmissionEvaluation(
            admitted=False,
            reason_code=MemoryProviderAdmissionReasonCode.DURABILITY_EVIDENCE_MISMATCH,
            provider_id=trusted_provider_id,
            declared_qualification_status=classification.declared_qualification_status,
            trusted_qualification_status=evidence.status,
            qualification_run_id=evidence.qualification_run_id,
            trusted_durability_status=durability_evidence.durability_status,
            durability_run_id=durability_evidence.qualification_run_id,
        )
    if (
        durability_evidence.durability_status
        is not MemoryProviderTrustedDurabilityStatus.DURABLE
    ):
        return UserProfileStoreProductionAdmissionEvaluation(
            admitted=False,
            reason_code=MemoryProviderAdmissionReasonCode.PROVIDER_NOT_DURABLE,
            provider_id=trusted_provider_id,
            declared_qualification_status=classification.declared_qualification_status,
            trusted_qualification_status=evidence.status,
            qualification_run_id=evidence.qualification_run_id,
            trusted_durability_status=durability_evidence.durability_status,
            durability_run_id=durability_evidence.qualification_run_id,
        )
    if durability_evidence.qualification_run_id != evidence.qualification_run_id:
        return UserProfileStoreProductionAdmissionEvaluation(
            admitted=False,
            reason_code=MemoryProviderAdmissionReasonCode.DURABILITY_EVIDENCE_MISMATCH,
            provider_id=trusted_provider_id,
            declared_qualification_status=classification.declared_qualification_status,
            trusted_qualification_status=evidence.status,
            qualification_run_id=evidence.qualification_run_id,
            trusted_durability_status=durability_evidence.durability_status,
            durability_run_id=durability_evidence.qualification_run_id,
        )
    return UserProfileStoreProductionAdmissionEvaluation(
        admitted=True,
        reason_code=None,
        provider_id=trusted_identity.provider_id,
        declared_qualification_status=classification.declared_qualification_status,
        trusted_qualification_status=evidence.status,
        qualification_run_id=evidence.qualification_run_id,
        trusted_durability_status=durability_evidence.durability_status,
        durability_run_id=durability_evidence.qualification_run_id,
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
        declared_qualification_status: MemoryProviderQualificationStatus,
        trusted_qualification_status: MemoryProviderQualificationStatus | None,
        reference_only: bool,
        qualification_run_id: str | None = None,
        declared_provider_id: str | None = None,
        trusted_provider_id: str | None = None,
        trusted_backing_provider_id: str | None = None,
        evidence_backing_provider_id: str | None = None,
        trusted_durability_status: MemoryProviderTrustedDurabilityStatus | None = None,
        durability_run_id: str | None = None,
    ) -> None:
        self.capability = capability
        self.execution_mode = execution_mode
        self.application_profile = application_profile
        self.reason_code = reason_code
        self.provider_id = provider_id
        self.durability = durability
        self.declared_qualification_status = declared_qualification_status
        self.trusted_qualification_status = trusted_qualification_status
        self.reference_only = reference_only
        self.qualification_run_id = qualification_run_id
        self.declared_provider_id = declared_provider_id
        self.trusted_provider_id = trusted_provider_id
        self.trusted_backing_provider_id = trusted_backing_provider_id
        self.evidence_backing_provider_id = evidence_backing_provider_id
        self.trusted_durability_status = trusted_durability_status
        self.durability_run_id = durability_run_id
        super().__init__(
            "memory provider admission failed: "
            f"capability={capability.value}; "
            f"execution_mode={execution_mode}; "
            f"application_profile={application_profile}; "
            f"reason_code={reason_code.value}; "
            f"provider_id={provider_id}; "
            f"declared_provider_id={declared_provider_id or provider_id}; "
            f"trusted_provider_id={trusted_provider_id or 'none'}; "
            f"backing_provider_id={trusted_backing_provider_id or 'none'}; "
            f"evidence_backing_provider_id={evidence_backing_provider_id or 'none'}; "
            f"declared_durability={durability.value}; "
            f"declared_qualification_status={declared_qualification_status.value}; "
            f"trusted_durability_status="
            f"{trusted_durability_status.value if trusted_durability_status else 'none'}; "
            f"durability_run_id={durability_run_id or 'none'}; "
            f"trusted_qualification_status="
            f"{trusted_qualification_status.value if trusted_qualification_status else 'none'}; "
            f"reference_only={reference_only}; "
            f"qualification_run_id={qualification_run_id or 'none'}",
        )
