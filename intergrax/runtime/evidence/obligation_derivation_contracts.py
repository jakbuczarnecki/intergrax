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

from enum import StrEnum
from typing import Annotated, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field


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


class DerivedIndexedEvidenceObligationV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    obligation_kind: Literal["indexed"] = "indexed"
    requirement_id: str = Field(..., min_length=1, max_length=128)
    semantic_role: str = Field(..., min_length=1, max_length=256)
    indexed_source_binding_id: str | None = Field(
        default=None, min_length=1, max_length=128
    )
    source_policy_document_id: str = Field(..., min_length=1, max_length=128)
    source_revision_id: str = Field(..., min_length=1, max_length=128)
    source_rule_id: str = Field(..., min_length=1, max_length=128)


class DerivedLiveEvidenceObligationV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    obligation_kind: Literal["live"] = "live"
    requirement_id: str = Field(..., min_length=1, max_length=128)
    semantic_role: str = Field(..., min_length=1, max_length=256)
    call_id: str = Field(..., min_length=1, max_length=128)
    source_policy_document_id: str = Field(..., min_length=1, max_length=128)
    source_revision_id: str = Field(..., min_length=1, max_length=128)
    source_rule_id: str = Field(..., min_length=1, max_length=128)


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
    source_policy_document_id: str = Field(..., min_length=1, max_length=128)
    source_revision_id: str = Field(..., min_length=1, max_length=128)
    source_rule_id: str = Field(..., min_length=1, max_length=128)


class DerivedEvidenceContractV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    derivation_snapshot_id: str = Field(..., min_length=1, max_length=256)
    source_policy_document_ids: tuple[str, ...] = ()
    source_revision_ids: tuple[str, ...] = ()
    source_rule_ids: tuple[str, ...] = ()
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
