# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Agent package trust and provenance evidence contracts (AGENT_DISTRIBUTION §10)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from intergrax.agent_distribution._digest import (
    content_digest_for_model,
    normalize_package_digest,
)
from intergrax.agent_distribution.catalog import CatalogProviderKind
from intergrax.agent_distribution.identity import AgentPackageIdentity
from intergrax.core.qualification import (
    QualificationEvidence,
    QualificationStatus,
)

_NON_EMPTY = Field(min_length=1)


def _strip_required(value: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError("must be non-empty")
    return normalized


def require_timezone_aware_utc_datetime(
    value: datetime,
    *,
    field_name: str,
) -> datetime:
    """Reject naive datetimes; normalize timezone-aware values to UTC."""
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware UTC datetime")
    if value.utcoffset() != timedelta(0):
        return value.astimezone(UTC)
    return value


class AgentDeliverySource(StrEnum):
    """How an agent package entered the distribution plane — audit only."""

    MARKETPLACE = "marketplace"
    ORG_REGISTRY = "org_registry"
    WORKSPACE = "workspace"
    BUILTIN = "builtin"
    AIRGAP_BUNDLE = "airgap_bundle"
    LOCAL_DEVELOPER = "local_developer"


class AgentQualificationEvidenceKind(StrEnum):
    """Auditable evidence categories for agent packages."""

    CONTRACT_VALIDATION = "contract_validation"
    PLATFORM_COMPATIBILITY = "platform_compatibility"
    SIGNATURE_VERIFICATION = "signature_verification"
    REVOCATION_CHECK = "revocation_check"
    ORG_POLICY_DECISION = "org_policy_decision"
    DOMAIN_QUALIFICATION = "domain_qualification"


class AgentPublisherIdentity(BaseModel):
    """Publisher identity reference — not execution subject identity."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    publisher_id: str = _NON_EMPTY
    display_name: str | None = None
    organization_id: str | None = None

    @field_validator("publisher_id", "display_name", "organization_id")
    @classmethod
    def _strip_optional(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _strip_required(value)


class AgentTrustEvidenceRef(BaseModel):
    """Opaque evidence reference carried on installation and revision records."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    evidence_id: str = _NON_EMPTY
    kind: AgentQualificationEvidenceKind
    ref: str | None = None

    @field_validator("evidence_id", "ref")
    @classmethod
    def _strip_optional(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _strip_required(value)


@dataclass(frozen=True, slots=True)
class AgentPackageQualificationResult:
    """Immutable qualification snapshot for audit and later trust coordinator."""

    publisher: AgentPublisherIdentity
    status: QualificationStatus
    evidence: tuple[QualificationEvidence[AgentQualificationEvidenceKind], ...]
    reason: str
    delivery_source: AgentDeliverySource
    qualified_at: datetime

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "qualified_at",
            require_timezone_aware_utc_datetime(
                self.qualified_at,
                field_name="qualified_at",
            ),
        )

    @property
    def production_allowed(self) -> bool:
        return self.status is QualificationStatus.PRODUCTION_QUALIFIED


class AgentInstallationTrustRecord(BaseModel):
    """Trust evidence references persisted on installation records (§10.3)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    trust_evidence_refs: tuple[AgentTrustEvidenceRef, ...] = ()
    qualification_status: QualificationStatus
    package_digest: str = _NON_EMPTY
    publisher_identity_ref: str = _NON_EMPTY
    source_provider_id: str = _NON_EMPTY
    source_entry_ref: str | None = None
    revocation_checked_at: datetime | None = None
    qualification_qualified_at: datetime | None = None
    org_policy_decision_ref: str | None = None
    policy_fingerprint: str | None = None

    @field_validator("revocation_checked_at", "qualification_qualified_at")
    @classmethod
    def _validate_utc_datetime(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        return require_timezone_aware_utc_datetime(value, field_name="datetime")

    @field_validator("package_digest")
    @classmethod
    def _validate_package_digest(cls, value: str) -> str:
        return normalize_package_digest(value)

    @field_validator(
        "publisher_identity_ref",
        "source_provider_id",
        "source_entry_ref",
        "org_policy_decision_ref",
        "policy_fingerprint",
    )
    @classmethod
    def _strip_optional(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _strip_required(value)


class AgentPackageTrustPosture(StrEnum):
    """Trust evaluation posture — behavior is driven by policy, not environment names."""

    DEVELOPMENT = "development"
    PRODUCTION = "production"


class AgentPackageTrustOutcome(StrEnum):
    """Canonical trust coordinator decision."""

    ALLOW = "allow"
    DENY = "deny"
    REVIEW = "review"


class AgentPackageTrustReasonCode(StrEnum):
    """Stable machine-readable trust decision codes."""

    QUALIFIED = "qualified"
    PACKAGE_DIGEST_MISMATCH = "package_digest_mismatch"
    PACKAGE_DIGEST_REVOKED = "package_digest_revoked"
    PUBLISHER_MISMATCH = "publisher_mismatch"
    PUBLISHER_DENIED = "publisher_denied"
    PUBLISHER_REVOKED = "publisher_revoked"
    SOURCE_NOT_PERMITTED = "source_not_permitted"
    SOURCE_DENIED = "source_denied"
    SOURCE_REVOKED = "source_revoked"
    SOURCE_DISABLED = "source_disabled"
    MISSING_REQUIRED_EVIDENCE = "missing_required_evidence"
    INSUFFICIENT_QUALIFICATION_STATUS = "insufficient_qualification_status"
    EVIDENCE_DIGEST_MISMATCH = "evidence_digest_mismatch"
    MISSING_PACKAGE_DIGEST_EVIDENCE = "missing_package_digest_evidence"
    EVIDENCE_PACKAGE_MISMATCH = "evidence_package_mismatch"
    EVIDENCE_REVOKED = "evidence_revoked"
    MALFORMED_EVIDENCE = "malformed_evidence"
    UNSIGNED_FORBIDDEN = "unsigned_forbidden"
    UNQUALIFIED_FORBIDDEN = "unqualified_forbidden"
    VERSION_LABEL_WITHOUT_DIGEST = "version_label_without_digest"
    QUALIFICATION_EXPIRED = "qualification_expired"
    QUALIFICATION_TIMESTAMP_INVALID = "qualification_timestamp_invalid"


class AgentPackageTrustPolicy(BaseModel):
    """Deterministic, serializable trust policy input for package qualification."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    posture: AgentPackageTrustPosture = AgentPackageTrustPosture.PRODUCTION
    trust_profile_ref: str | None = None
    permitted_provider_kinds: frozenset[CatalogProviderKind] | None = None
    permitted_catalog_source_ids: frozenset[str] | None = None
    permitted_delivery_sources: frozenset[AgentDeliverySource] | None = None
    permitted_publisher_ids: frozenset[str] | None = None
    denied_publisher_ids: frozenset[str] = frozenset()
    denied_catalog_source_ids: frozenset[str] = frozenset()
    denied_package_digests: frozenset[str] = frozenset()
    required_qualification_status: QualificationStatus | None = None
    required_evidence_kinds: frozenset[AgentQualificationEvidenceKind] = frozenset()
    forbid_unsigned_or_unqualified: bool = True
    max_qualification_age: timedelta | None = None

    @field_validator("max_qualification_age")
    @classmethod
    def _validate_max_qualification_age(
        cls,
        value: timedelta | None,
    ) -> timedelta | None:
        if value is None:
            return None
        if value <= timedelta(0):
            raise ValueError("max_qualification_age must be positive when set")
        return value

    @field_validator("trust_profile_ref")
    @classmethod
    def _strip_optional(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _strip_required(value)

    @property
    def effective_required_qualification_status(self) -> QualificationStatus:
        if self.required_qualification_status is not None:
            return self.required_qualification_status
        if self.posture is AgentPackageTrustPosture.PRODUCTION:
            return QualificationStatus.PRODUCTION_QUALIFIED
        return QualificationStatus.QUALIFIED

    @property
    def policy_fingerprint(self) -> str:
        """Deterministic identity for the policy value used at admission time."""
        return content_digest_for_model(self)


class AgentPackageTrustRevocationState(BaseModel):
    """Authoritative revocation state supplied to trust evaluation (no network fetch)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    revoked_publisher_ids: frozenset[str] = frozenset()
    revoked_package_digests: frozenset[str] = frozenset()
    revoked_evidence_ids: frozenset[str] = frozenset()
    revoked_catalog_source_ids: frozenset[str] = frozenset()
    disabled_catalog_source_ids: frozenset[str] = frozenset()


@dataclass(frozen=True, slots=True)
class AgentPackageTrustDecision:
    """Immutable trust evaluation outcome with audit evidence."""

    outcome: AgentPackageTrustOutcome
    reason_code: AgentPackageTrustReasonCode
    reason: str
    package_identity: AgentPackageIdentity
    publisher: AgentPublisherIdentity
    catalog_source_id: str
    delivery_source: AgentDeliverySource
    policy_profile_ref: str | None
    qualification: AgentPackageQualificationResult | None
    trust_record: AgentInstallationTrustRecord | None
    trust_evidence_refs: tuple[AgentTrustEvidenceRef, ...]

    @property
    def installable(self) -> bool:
        return (
            self.outcome is AgentPackageTrustOutcome.ALLOW
            and self.trust_record is not None
        )

    def to_audit_dict(self) -> dict[str, Any]:
        """Deterministic audit payload — stable across repeated evaluation."""
        return {
            "outcome": self.outcome.value,
            "reason_code": self.reason_code.value,
            "reason": self.reason,
            "package_digest": self.package_identity.package_digest,
            "distribution_package_id": self.package_identity.distribution_package_id,
            "publisher_id": self.publisher.publisher_id,
            "catalog_source_id": self.catalog_source_id,
            "delivery_source": self.delivery_source.value,
            "policy_profile_ref": self.policy_profile_ref,
            "policy_fingerprint": (
                self.trust_record.policy_fingerprint
                if self.trust_record is not None
                else None
            ),
            "qualification_status": (
                self.qualification.status.value
                if self.qualification is not None
                else None
            ),
            "qualification_qualified_at": (
                self.qualification.qualified_at.isoformat()
                if self.qualification is not None
                else (
                    self.trust_record.qualification_qualified_at.isoformat()
                    if self.trust_record is not None
                    and self.trust_record.qualification_qualified_at is not None
                    else None
                )
            ),
            "revocation_checked_at": (
                self.trust_record.revocation_checked_at.isoformat()
                if self.trust_record is not None
                and self.trust_record.revocation_checked_at is not None
                else None
            ),
            "trust_evidence_refs": [
                {
                    "evidence_id": ref.evidence_id,
                    "kind": ref.kind.value,
                    "ref": ref.ref,
                }
                for ref in self.trust_evidence_refs
            ],
        }
