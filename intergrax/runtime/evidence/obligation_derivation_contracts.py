# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

"""Typed contracts for deterministic evidence obligation derivation (COMM-5F3-A).

F3-A operates on already-resolved typed policy rules. Natural-language policy
interpretation and rule resolution happen upstream of this boundary.

Architectural ownership (F3-A):
- Derivation emits authoritative indexed obligations and, for live rules, both
  deterministic live-call proposals and matching live obligations so every
  live obligation references an existing planned call identity.
- Downstream planning composes product, policy-derived, provider, and caller
  obligations additively; caller layers cannot remove earlier authority.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Annotated, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_MAX_TEMPORAL_AGE_SECONDS = 31_536_000


def _require_timezone_aware(value: datetime, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name}_must_be_timezone_aware")
    return value


class MaxAgeTemporalConstraintV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["max_age"] = "max_age"
    max_age_seconds: int = Field(..., ge=1, le=_MAX_TEMPORAL_AGE_SECONDS)


class ValidAtTemporalConstraintV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["valid_at"] = "valid_at"


TemporalConstraintV1 = Annotated[
    MaxAgeTemporalConstraintV1 | ValidAtTemporalConstraintV1,
    Field(discriminator="kind"),
]


class PointInTimeEvidenceTemporalV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["point_in_time"] = "point_in_time"
    effective_at: datetime

    @field_validator("effective_at")
    @classmethod
    def _timezone_aware(cls, value: datetime) -> datetime:
        return _require_timezone_aware(value, "effective_at")


class ValidityIntervalEvidenceTemporalV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["validity_interval"] = "validity_interval"
    valid_from: datetime
    valid_until: datetime

    @field_validator("valid_from", "valid_until")
    @classmethod
    def _timezone_aware(cls, value: datetime) -> datetime:
        return _require_timezone_aware(value, "valid_from")

    @model_validator(mode="after")
    def _validate_interval(self) -> ValidityIntervalEvidenceTemporalV1:
        if self.valid_until < self.valid_from:
            raise ValueError("validity_interval_invalid")
        return self


EvidenceTemporalMetadataV1 = Annotated[
    PointInTimeEvidenceTemporalV1 | ValidityIntervalEvidenceTemporalV1,
    Field(discriminator="kind"),
]


class EvidenceObligationDerivationError(RuntimeError):
    """Fail-closed derivation validation error with a stable code."""

    def __init__(self, error_code: str) -> None:
        super().__init__(error_code)
        self.error_code = error_code


class ResolvedPolicyRuleKindV1(StrEnum):
    REQUIRE_INDEXED_EVIDENCE = "require_indexed_evidence"
    REQUIRE_LIVE_EVIDENCE = "require_live_evidence"


class RequireIndexedEvidenceRuleParametersV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    semantic_role: str = Field(..., min_length=1, max_length=256)
    requirement_key: str = Field(..., min_length=1, max_length=128)
    indexed_source_binding_id: str | None = Field(
        default=None, min_length=1, max_length=128
    )
    temporal_constraint: TemporalConstraintV1 | None = None


class TypedCapabilityRequestEntryV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    key: str = Field(..., min_length=1, max_length=128)
    value: str = Field(..., max_length=4096)


class RequireLiveEvidenceRuleParametersV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    semantic_role: str = Field(..., min_length=1, max_length=256)
    requirement_key: str = Field(..., min_length=1, max_length=128)
    capability_id: str = Field(..., min_length=1, max_length=128)
    live_access_binding_id: str = Field(..., min_length=1, max_length=128)
    live_call_descriptor_ref: str = Field(..., min_length=1, max_length=128)
    typed_capability_request: tuple[TypedCapabilityRequestEntryV1, ...] = ()
    temporal_constraint: TemporalConstraintV1 | None = None


class RequireIndexedEvidencePolicyRuleV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    rule_kind: Literal[ResolvedPolicyRuleKindV1.REQUIRE_INDEXED_EVIDENCE] = (
        ResolvedPolicyRuleKindV1.REQUIRE_INDEXED_EVIDENCE
    )
    policy_document_id: str = Field(..., min_length=1, max_length=128)
    revision_id: str = Field(..., min_length=1, max_length=128)
    rule_id: str = Field(..., min_length=1, max_length=128)
    parameters: RequireIndexedEvidenceRuleParametersV1


class RequireLiveEvidencePolicyRuleV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    rule_kind: Literal[ResolvedPolicyRuleKindV1.REQUIRE_LIVE_EVIDENCE] = (
        ResolvedPolicyRuleKindV1.REQUIRE_LIVE_EVIDENCE
    )
    policy_document_id: str = Field(..., min_length=1, max_length=128)
    revision_id: str = Field(..., min_length=1, max_length=128)
    rule_id: str = Field(..., min_length=1, max_length=128)
    parameters: RequireLiveEvidenceRuleParametersV1


ResolvedPolicyRuleV1 = Annotated[
    RequireIndexedEvidencePolicyRuleV1 | RequireLiveEvidencePolicyRuleV1,
    Field(discriminator="rule_kind"),
]


class EvidenceObligationDerivationContextV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    tenant_id: str = Field(..., min_length=1, max_length=128)
    workspace_id: str = Field(..., min_length=1, max_length=128)
    configuration_revision: int = Field(..., ge=0)
    resolved_policy_rules: tuple[ResolvedPolicyRuleV1, ...] = ()


class RequirementOriginV1(BaseModel):
    """Structural provenance for one policy-derived evidence obligation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    policy_document_id: str = Field(..., min_length=1, max_length=128)
    revision_id: str = Field(..., min_length=1, max_length=128)
    rule_id: str = Field(..., min_length=1, max_length=128)


class PolicyRevisionReferenceV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    policy_document_id: str = Field(..., min_length=1, max_length=128)
    revision_id: str = Field(..., min_length=1, max_length=128)


class PolicyEvidenceBasisV1(BaseModel):
    """Authoritative policy revisions governing one derivation/plan/run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    policy_revisions: tuple[PolicyRevisionReferenceV1, ...]
    derivation_snapshot_id: str = Field(..., min_length=1, max_length=256)

    @model_validator(mode="after")
    def _validate_canonical_basis(self) -> PolicyEvidenceBasisV1:
        seen_documents: set[str] = set()
        previous_document_id: str | None = None
        for reference in self.policy_revisions:
            if reference.policy_document_id in seen_documents:
                raise ValueError("duplicate_policy_document_in_basis")
            seen_documents.add(reference.policy_document_id)
            if (
                previous_document_id is not None
                and reference.policy_document_id < previous_document_id
            ):
                raise ValueError("policy_revisions_not_canonically_ordered")
            previous_document_id = reference.policy_document_id
        return self


class DerivedIndexedEvidenceObligationV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    obligation_kind: Literal["indexed"] = "indexed"
    requirement_id: str = Field(..., min_length=1, max_length=128)
    semantic_role: str = Field(..., min_length=1, max_length=256)
    indexed_source_binding_id: str | None = Field(
        default=None, min_length=1, max_length=128
    )
    temporal_constraint: TemporalConstraintV1 | None = None
    origin: RequirementOriginV1


class DerivedLiveEvidenceObligationV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    obligation_kind: Literal["live"] = "live"
    requirement_id: str = Field(..., min_length=1, max_length=128)
    semantic_role: str = Field(..., min_length=1, max_length=256)
    call_id: str = Field(..., min_length=1, max_length=128)
    temporal_constraint: TemporalConstraintV1 | None = None
    origin: RequirementOriginV1


DerivedEvidenceObligationV1 = Annotated[
    DerivedIndexedEvidenceObligationV1 | DerivedLiveEvidenceObligationV1,
    Field(discriminator="obligation_kind"),
]


class DerivedLiveCallProposalV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    call_id: str = Field(..., min_length=1, max_length=128)
    live_access_binding_id: str = Field(..., min_length=1, max_length=128)
    capability_id: str = Field(..., min_length=1, max_length=128)
    typed_capability_request: tuple[TypedCapabilityRequestEntryV1, ...] = ()
    origin: RequirementOriginV1


class DerivedEvidenceContractV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    derivation_snapshot_id: str = Field(..., min_length=1, max_length=256)
    policy_basis: PolicyEvidenceBasisV1 | None = None
    derived_obligations: tuple[DerivedEvidenceObligationV1, ...] = ()
    derived_live_call_proposals: tuple[DerivedLiveCallProposalV1, ...] = ()


@runtime_checkable
class EvidenceObligationDerivationPort(Protocol):
  """Provider-neutral, side-effect-free obligation derivation."""

  def derive(
      self,
      context: EvidenceObligationDerivationContextV1,
  ) -> DerivedEvidenceContractV1:
      ...
