# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Governance narrowing contracts for capability discovery (Stage 5)."""

from __future__ import annotations

from enum import StrEnum
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from intergrax.contracts.capability_catalog._validation import require_non_empty_text
from intergrax.contracts.capability_catalog.identity_key import CapabilityIdentityKey

SCHEMA_GOVERNANCE_DECISION_EVIDENCE_V1: Final = "governance_decision_evidence.v1"
SCHEMA_CAPABILITY_GOVERNANCE_CONTEXT_V1: Final = "capability_governance_context.v1"
SCHEMA_CAPABILITY_TOOL_GOVERNANCE_EVIDENCE_V1: Final = (
    "capability_tool_governance_evidence.v1"
)
SCHEMA_CAPABILITY_AGENT_GOVERNANCE_EVIDENCE_V1: Final = (
    "capability_agent_governance_evidence.v1"
)
SCHEMA_CAPABILITY_SKILL_GOVERNANCE_EVIDENCE_V1: Final = (
    "capability_skill_governance_evidence.v1"
)

_NON_EMPTY = Field(min_length=1)


class GovernanceDisposition(StrEnum):
    """Narrowing outcome — projection only, not domain authority."""

    ALLOWED = "allowed"
    BLOCKED = "blocked"


class CapabilityGovernancePosture(StrEnum):
    """Fail-closed posture for optional evidence requirements."""

    STRICT = "strict"
    NON_STRICT = "non_strict"


class CapabilityGovernanceReasonCode(StrEnum):
    """Stable machine-readable governance reason codes."""

    AVAILABILITY_BLOCKED = "availability_blocked"
    AVAILABILITY_UNAVAILABLE = "availability_unavailable"
    AVAILABILITY_SCOPE_UNAVAILABLE = "availability_scope_unavailable"
    POLICY_DENIED = "policy_denied"
    AUTHORITY_INSUFFICIENT = "authority_insufficient"
    TRUST_NOT_SATISFIED = "trust_not_satisfied"
    NOT_ENTITLED = "not_entitled"
    SCOPE_FORBIDDEN = "scope_forbidden"
    MISSING_REQUIRED_EVIDENCE = "missing_required_evidence"
    CONFLICTING_GOVERNANCE_EVIDENCE = "conflicting_governance_evidence"
    EVALUATOR_FAILURE = "evaluator_failure"
    EVALUATOR_INVALID_OUTPUT = "evaluator_invalid_output"
    GOVERNANCE_ALLOWED = "governance_allowed"
    GOVERNANCE_NOT_APPLICABLE = "governance_not_applicable"


NORMATIVE_CAPABILITY_GOVERNANCE_REASON_CODES: Final[
    frozenset[CapabilityGovernanceReasonCode]
] = frozenset(CapabilityGovernanceReasonCode)


class GovernanceDecisionEvidence(BaseModel):
    """Typed audit evidence for a single evaluator decision."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["governance_decision_evidence.v1"] = (
        SCHEMA_GOVERNANCE_DECISION_EVIDENCE_V1
    )
    evaluator_id: str = _NON_EMPTY
    disposition: GovernanceDisposition
    reason_code: CapabilityGovernanceReasonCode
    detail: str | None = None
    reference: str | None = None

    @field_validator("evaluator_id")
    @classmethod
    def _validate_evaluator_id(cls, value: str) -> str:
        return require_non_empty_text(value, label="evaluator_id")


class CapabilityToolGovernanceEvidence(BaseModel):
    """Read-only Tool access/policy projection — caller supplied."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["capability_tool_governance_evidence.v1"] = (
        SCHEMA_CAPABILITY_TOOL_GOVERNANCE_EVIDENCE_V1
    )
    allowed_keys: tuple[CapabilityIdentityKey, ...] = ()
    denied_keys: tuple[CapabilityIdentityKey, ...] = ()

    @model_validator(mode="after")
    def _validate_disjoint_keys(
        self,
    ) -> CapabilityToolGovernanceEvidence:
        allowed = {key.sort_key for key in self.allowed_keys}
        denied = {key.sort_key for key in self.denied_keys}
        overlap = allowed & denied
        if overlap:
            conflict_key = min(overlap)
            raise ValueError(
                "tool governance evidence conflict: identity "
                f"{conflict_key!r} appears in both allowed_keys and denied_keys",
            )
        return self


class CapabilityAgentGovernanceEvidence(BaseModel):
    """Read-only Agent trust/admission projection — caller supplied."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["capability_agent_governance_evidence.v1"] = (
        SCHEMA_CAPABILITY_AGENT_GOVERNANCE_EVIDENCE_V1
    )
    trusted_keys: tuple[CapabilityIdentityKey, ...] = ()
    blocked_keys: tuple[CapabilityIdentityKey, ...] = ()
    revoked_keys: tuple[CapabilityIdentityKey, ...] = ()

    @model_validator(mode="after")
    def _validate_disjoint_keys(
        self,
    ) -> CapabilityAgentGovernanceEvidence:
        trusted = {key.sort_key for key in self.trusted_keys}
        blocked = {key.sort_key for key in self.blocked_keys}
        revoked = {key.sort_key for key in self.revoked_keys}
        for left_label, right_label, left, right in (
            ("trusted_keys", "blocked_keys", trusted, blocked),
            ("trusted_keys", "revoked_keys", trusted, revoked),
            ("blocked_keys", "revoked_keys", blocked, revoked),
        ):
            overlap = left & right
            if overlap:
                conflict_key = min(overlap)
                raise ValueError(
                    "agent governance evidence conflict: identity "
                    f"{conflict_key!r} appears in both {left_label} and "
                    f"{right_label}",
                )
        return self


class CapabilitySkillGovernanceEvidence(BaseModel):
    """Read-only Skill profile enablement projection — caller supplied."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["capability_skill_governance_evidence.v1"] = (
        SCHEMA_CAPABILITY_SKILL_GOVERNANCE_EVIDENCE_V1
    )
    enabled_keys: tuple[CapabilityIdentityKey, ...] = ()
    blocked_keys: tuple[CapabilityIdentityKey, ...] = ()

    @model_validator(mode="after")
    def _validate_disjoint_keys(
        self,
    ) -> CapabilitySkillGovernanceEvidence:
        enabled = {key.sort_key for key in self.enabled_keys}
        blocked = {key.sort_key for key in self.blocked_keys}
        overlap = enabled & blocked
        if overlap:
            conflict_key = min(overlap)
            raise ValueError(
                "skill governance evidence conflict: identity "
                f"{conflict_key!r} appears in both enabled_keys and blocked_keys",
            )
        return self


class CapabilityGovernanceContext(BaseModel):
    """Read-only governance facts — no mutable registries or authority grants."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["capability_governance_context.v1"] = (
        SCHEMA_CAPABILITY_GOVERNANCE_CONTEXT_V1
    )
    posture: CapabilityGovernancePosture = CapabilityGovernancePosture.STRICT
    tool_evidence: CapabilityToolGovernanceEvidence | None = None
    agent_evidence: CapabilityAgentGovernanceEvidence | None = None
    skill_evidence: CapabilitySkillGovernanceEvidence | None = None
