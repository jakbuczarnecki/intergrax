# © Artur Czarnecki. All rights reserved.

"""Physical delegation governance contracts (NPSC-5D/R2).

Post-selection, pre-acquisition authorization for one selected specialist identity.
Evaluated through canonical ``PolicyDecision`` / ``PolicyAction`` — not a second engine.
"""

from __future__ import annotations

from typing import Final, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator

from intergrax.contracts.agent_run import RequestIdentity
from intergrax.contracts.control_plane_mutation import GovernanceEvaluationPoint
from intergrax.contracts.evaluated_policy_decision import request_digest_for_payload
from intergrax.contracts.runtime_policy import PolicyAction, PolicyDecision

SCHEMA_PHYSICAL_DELEGATION_GOVERNANCE_REQUEST_V1: Final = (
    "physical_delegation_governance_request.v1"
)
SCHEMA_PHYSICAL_DELEGATION_GOVERNANCE_EVIDENCE_V1: Final = (
    "physical_delegation_governance_evidence.v1"
)

_NON_EMPTY = Field(min_length=1)


class PhysicalDelegationSelectedIdentity(BaseModel):
    """Governance-owned projection of canonical selected specialist identity."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    catalog_source_id: str = _NON_EMPTY
    provider_kind: str = _NON_EMPTY
    distribution_package_id: str = _NON_EMPTY
    package_version: str = _NON_EMPTY
    package_digest: str | None = None

    @field_validator(
        "catalog_source_id",
        "provider_kind",
        "distribution_package_id",
        "package_version",
    )
    @classmethod
    def _strip_required(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("must be non-empty")
        return normalized


class PhysicalDelegationCapabilityRequirement(BaseModel):
    """Governance-owned projection of canonical capability requirement."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    required_capability_ids: tuple[str, ...] = Field(min_length=1)

    @field_validator("required_capability_ids")
    @classmethod
    def _normalize_capability_ids(
        cls,
        value: tuple[str, ...],
    ) -> tuple[str, ...]:
        normalized = tuple(
            sorted({item.strip() for item in value if item.strip()}),
        )
        if not normalized:
            raise ValueError("required_capability_ids must be non-empty")
        return normalized


class PhysicalDelegationGovernanceRequest(BaseModel):
    """Stage-2 physical delegation admission — exact selected specialist identity."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["physical_delegation_governance_request.v1"] = (
        SCHEMA_PHYSICAL_DELEGATION_GOVERNANCE_REQUEST_V1
    )
    evaluation_point: GovernanceEvaluationPoint = (
        GovernanceEvaluationPoint.MULTI_AGENT_DELEGATION
    )
    delegation_id: str = _NON_EMPTY
    task_scope_id: str = _NON_EMPTY
    application_id: str = _NON_EMPTY
    application_environment_id: str = _NON_EMPTY
    principal: RequestIdentity
    capability_requirement: PhysicalDelegationCapabilityRequirement
    selected_identity: PhysicalDelegationSelectedIdentity
    requested_permission_scopes: tuple[str, ...] = Field(default_factory=tuple)

    @field_validator(
        "delegation_id",
        "task_scope_id",
        "application_id",
        "application_environment_id",
    )
    @classmethod
    def _strip_required(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("must be non-empty")
        return normalized

    @field_validator("requested_permission_scopes")
    @classmethod
    def _normalize_permission_scopes(
        cls,
        value: tuple[str, ...],
    ) -> tuple[str, ...]:
        return tuple(scope.strip() for scope in value if scope.strip())

    @property
    def tenant_id(self) -> str:
        return self.principal.tenant_id


class PhysicalDelegationGovernanceEvidence(BaseModel):
    """Typed authorization provenance for physical delegation admission."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["physical_delegation_governance_evidence.v1"] = (
        SCHEMA_PHYSICAL_DELEGATION_GOVERNANCE_EVIDENCE_V1
    )
    evaluation_point: GovernanceEvaluationPoint = (
        GovernanceEvaluationPoint.MULTI_AGENT_DELEGATION
    )
    request_digest: str = _NON_EMPTY
    delegation_id: str = _NON_EMPTY
    tenant_id: str = _NON_EMPTY
    task_scope_id: str = _NON_EMPTY
    application_id: str = _NON_EMPTY
    application_environment_id: str = _NON_EMPTY
    selected_package_id: str = _NON_EMPTY
    policy_action: PolicyAction
    policy_rule_id: str = ""
    policy_decision_id: str = ""

    @field_validator("request_digest")
    @classmethod
    def _validate_request_digest(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized.startswith("sha256:"):
            raise ValueError("request_digest_must_be_sha256")
        return normalized


class PhysicalDelegationGovernanceResult(BaseModel):
    """Evaluation-only physical delegation admission outcome."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    permitted: bool
    decision: PolicyDecision
    evidence: PhysicalDelegationGovernanceEvidence
    requires_governed_continuation: bool = False
    validation_failed: bool = False


class PhysicalDelegationGovernancePolicyRule:
    """Immutable runtime rule for physical delegation admission evaluation."""

    __slots__ = ("rule_id", "decision", "package_id", "reason")

    def __init__(
        self,
        *,
        rule_id: str,
        decision: PolicyAction,
        package_id: str | None = None,
        reason: str = "",
    ) -> None:
        normalized_rule_id = rule_id.strip()
        if not normalized_rule_id:
            raise ValueError("rule_id must be non-empty")
        if not isinstance(decision, PolicyAction):
            raise TypeError("decision must be PolicyAction")
        normalized_package_id = package_id.strip() if package_id is not None else None
        if package_id is not None and not normalized_package_id:
            raise ValueError("package_id must be non-empty when provided")
        self.rule_id = normalized_rule_id
        self.decision = decision
        self.package_id = normalized_package_id
        self.reason = reason.strip()


class PhysicalDelegationGovernanceEvaluator(Protocol):
    """Configured policy evaluator for physical delegation admission."""

    def evaluate(
        self,
        request: PhysicalDelegationGovernanceRequest,
    ) -> PolicyDecision:
        """Return a fresh governance decision for ``request``."""
        ...


class PhysicalDelegationGovernancePort(Protocol):
    """Public governance admission boundary for physical delegation callers."""

    def evaluate(
        self,
        request: PhysicalDelegationGovernanceRequest,
    ) -> PhysicalDelegationGovernanceResult:
        """Evaluate physical delegation admission using typed request facts only."""
        ...


def physical_delegation_governance_request_digest(
    request: PhysicalDelegationGovernanceRequest,
) -> str:
    payload = request.model_dump(mode="json")
    return request_digest_for_payload(payload)


def evidence_from_request_and_decision(
    request: PhysicalDelegationGovernanceRequest,
    *,
    decision: PolicyDecision,
    request_digest: str,
) -> PhysicalDelegationGovernanceEvidence:
    return PhysicalDelegationGovernanceEvidence(
        request_digest=request_digest,
        delegation_id=request.delegation_id,
        tenant_id=request.tenant_id,
        task_scope_id=request.task_scope_id,
        application_id=request.application_id,
        application_environment_id=request.application_environment_id,
        selected_package_id=request.selected_identity.distribution_package_id,
        policy_action=decision.action,
        policy_rule_id=decision.policy_rule_id,
        policy_decision_id=decision.decision_id,
    )
