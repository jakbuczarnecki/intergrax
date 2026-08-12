# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Agent package trust and provenance evidence contracts (AGENT_DISTRIBUTION §10)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator

_NON_EMPTY = Field(min_length=1)


def _strip_required(value: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError("must be non-empty")
    return normalized


class AgentDeliverySource(StrEnum):
    """How an agent package entered the distribution plane — audit only."""

    MARKETPLACE = "marketplace"
    ORG_REGISTRY = "org_registry"
    WORKSPACE = "workspace"
    BUILTIN = "builtin"
    AIRGAP_BUNDLE = "airgap_bundle"
    LOCAL_DEVELOPER = "local_developer"


class AgentQualificationStatus(StrEnum):
    """Qualification outcome — distinct from installation lifecycle."""

    NOT_QUALIFIED = "not_qualified"
    QUALIFIED = "qualified"
    PRODUCTION_QUALIFIED = "production_qualified"
    REJECTED = "rejected"


class AgentQualificationEvidenceKind(StrEnum):
    """Auditable evidence categories for agent packages."""

    CONTRACT_VALIDATION = "contract_validation"
    PLATFORM_COMPATIBILITY = "platform_compatibility"
    SIGNATURE_VERIFICATION = "signature_verification"
    REVOCATION_CHECK = "revocation_check"
    ORG_POLICY_DECISION = "org_policy_decision"
    DOMAIN_QUALIFICATION = "domain_qualification"


@dataclass(frozen=True, slots=True)
class AgentQualificationEvidence:
    """Safe, immutable evidence metadata (no secrets or raw payloads)."""

    kind: AgentQualificationEvidenceKind
    code: str
    ref: str | None = None
    label: str | None = None


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
    """Immutable qualification record for audit and later trust coordinator."""

    publisher: AgentPublisherIdentity
    status: AgentQualificationStatus
    evidence: tuple[AgentQualificationEvidence, ...]
    reason: str
    delivery_source: AgentDeliverySource

    @property
    def production_allowed(self) -> bool:
        return self.status is AgentQualificationStatus.PRODUCTION_QUALIFIED


class AgentInstallationTrustRecord(BaseModel):
    """Trust evidence references persisted on installation records (§10.3)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    trust_evidence_refs: tuple[AgentTrustEvidenceRef, ...] = ()
    qualification_status: AgentQualificationStatus
    publisher_identity_ref: str = _NON_EMPTY
    source_provider_id: str = _NON_EMPTY
    source_entry_ref: str | None = None
    revocation_checked_at: datetime | None = None
    org_policy_decision_ref: str | None = None

    @field_validator("publisher_identity_ref", "source_provider_id", "source_entry_ref", "org_policy_decision_ref")
    @classmethod
    def _strip_optional(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _strip_required(value)
