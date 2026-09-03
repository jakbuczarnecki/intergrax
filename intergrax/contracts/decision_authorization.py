# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Decision governance authorization contracts (DS-GOV-01).

Canonical Decision-owned semantics for authorizing execution of an exact
authoritative Decision version for one explicit action under one policy context.
Distinct from verification quality, human review judgment, and declarative
tool-side HITL invocation grants.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Generic, NewType, Protocol, TypeVar, runtime_checkable
from uuid import uuid4

from intergrax.contracts.decision_identity import (
    DecisionIdentity,
    validate_decision_tenant_id,
)
from intergrax.contracts.decision_record import (
    AuthoritativeAcceptedDecision,
    DecisionLineageRef,
    DecisionProposalRef,
    decision_proposal_ref_sort_key,
)
from intergrax.contracts.decision_revision import proposal_refs_match

DecisionExecutionAuthorizationId = NewType("DecisionExecutionAuthorizationId", str)
DecisionExecutionActionKind = NewType("DecisionExecutionActionKind", str)

_AUTHORIZATION_ID_PREFIX = "deauth_"
_ACTION_KIND_PATTERN = re.compile(r"^[a-z][a-z0-9_.]{0,63}$")


def validate_decision_execution_authorization_id(
    value: object,
) -> DecisionExecutionAuthorizationId:
    if type(value) is not str:
        raise TypeError(
            "DecisionExecutionAuthorizationId must be str, "
            f"got {type(value).__name__}",
        )
    if not value or not value.strip():
        raise ValueError(
            "DecisionExecutionAuthorizationId must be non-empty and not whitespace-only",
        )
    if value != value.strip():
        raise ValueError(
            "DecisionExecutionAuthorizationId must not contain leading or trailing whitespace",
        )
    if not value.startswith(_AUTHORIZATION_ID_PREFIX):
        raise ValueError(
            "DecisionExecutionAuthorizationId must start with "
            f"{_AUTHORIZATION_ID_PREFIX!r}",
        )
    return DecisionExecutionAuthorizationId(value)


def mint_decision_execution_authorization_id() -> DecisionExecutionAuthorizationId:
    return DecisionExecutionAuthorizationId(f"{_AUTHORIZATION_ID_PREFIX}{uuid4().hex}")


def validate_decision_execution_action_kind(
    value: object,
) -> DecisionExecutionActionKind:
    if type(value) is not str:
        raise TypeError(
            f"DecisionExecutionActionKind must be str, got {type(value).__name__}",
        )
    if not value or not value.strip():
        raise ValueError(
            "DecisionExecutionActionKind must be non-empty and not whitespace-only",
        )
    if value != value.strip():
        raise ValueError(
            "DecisionExecutionActionKind must not contain leading or trailing whitespace",
        )
    if not _ACTION_KIND_PATTERN.fullmatch(value):
        raise ValueError(
            "DecisionExecutionActionKind must match [a-z][a-z0-9_.]{0,63}",
        )
    return DecisionExecutionActionKind(value)


def _validate_action_subject(value: object, label: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{label} must be str, got {type(value).__name__}")
    if not value or not value.strip():
        raise ValueError(f"{label} must be non-empty and not whitespace-only")
    if value != value.strip():
        raise ValueError(f"{label} must not contain leading or trailing whitespace")
    return value


@dataclass(frozen=True, slots=True)
class DecisionExecutionAction:
    """Domain-neutral execution intent bound to one action class."""

    kind: DecisionExecutionActionKind
    subject: str

    def __post_init__(self) -> None:
        validate_decision_execution_action_kind(self.kind)
        _validate_action_subject(self.subject, "DecisionExecutionAction.subject")


def decision_execution_action(
    *,
    kind: DecisionExecutionActionKind | str,
    subject: str,
) -> DecisionExecutionAction:
    """Build one typed execution action reference."""
    resolved_kind = (
        kind
        if type(kind) is DecisionExecutionActionKind
        else validate_decision_execution_action_kind(kind)
    )
    return DecisionExecutionAction(kind=resolved_kind, subject=subject)


@dataclass(frozen=True, slots=True)
class DecisionGovernancePolicyContext:
    """Exact governance policy provenance bound to one authorization decision."""

    policy_provenance_digest: str
    matched_rule_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if (
            type(self.policy_provenance_digest) is not str
            or not self.policy_provenance_digest.strip()
        ):
            raise ValueError(
                "DecisionGovernancePolicyContext.policy_provenance_digest must be non-empty",
            )
        for rule_id in self.matched_rule_ids:
            if type(rule_id) is not str or not rule_id.strip():
                raise ValueError(
                    "DecisionGovernancePolicyContext.matched_rule_ids "
                    "must contain non-empty str values",
                )


def decision_governance_policy_context(
    *,
    policy_provenance_digest: str,
    matched_rule_ids: tuple[str, ...] = (),
) -> DecisionGovernancePolicyContext:
    """Build one immutable governance policy context."""
    return DecisionGovernancePolicyContext(
        policy_provenance_digest=policy_provenance_digest,
        matched_rule_ids=matched_rule_ids,
    )


@dataclass(frozen=True, slots=True)
class AuthoritativeDecisionRef:
    """Exact authoritative accepted decision version reference."""

    identity: DecisionIdentity
    lineage_ref: DecisionLineageRef

    def __post_init__(self) -> None:
        if type(self.identity) is not DecisionIdentity:
            raise TypeError("AuthoritativeDecisionRef.identity must be DecisionIdentity")
        if type(self.lineage_ref) is not DecisionLineageRef:
            raise TypeError(
                "AuthoritativeDecisionRef.lineage_ref must be DecisionLineageRef",
            )
        if self.identity.version != self.lineage_ref.version:
            raise ValueError(
                "AuthoritativeDecisionRef identity.version must match lineage_ref.version",
            )


T = TypeVar("T")


def authoritative_decision_ref(
    decision: AuthoritativeAcceptedDecision[T],
) -> AuthoritativeDecisionRef:
    """Derive exact authoritative decision reference from accepted decision."""
    if type(decision) is not AuthoritativeAcceptedDecision:
        raise TypeError("decision must be AuthoritativeAcceptedDecision")
    return AuthoritativeDecisionRef(
        identity=decision.identity,
        lineage_ref=decision.lineage.current,
    )


def _authoritative_ref_as_proposal_ref(ref: AuthoritativeDecisionRef) -> DecisionProposalRef:
    return DecisionProposalRef(identity=ref.identity, lineage_ref=ref.lineage_ref)


def authoritative_decision_refs_match(
    left: AuthoritativeDecisionRef,
    right: AuthoritativeDecisionRef,
) -> bool:
    """Return whether two authoritative decision refs denote the same exact version."""
    return proposal_refs_match(
        _authoritative_ref_as_proposal_ref(left),
        _authoritative_ref_as_proposal_ref(right),
    )


def authoritative_decision_ref_sort_key(
    ref: AuthoritativeDecisionRef,
) -> tuple[str, str, str, str, str, str, str, int, str, int, str]:
    """Deterministic ordering key for authoritative decision references."""
    return decision_proposal_ref_sort_key(_authoritative_ref_as_proposal_ref(ref))


class DecisionGovernanceDisposition(str, Enum):
    """Governance evaluation outcome for one authoritative decision action."""

    ALLOW = "allow"
    DENY = "deny"
    REQUIRE_HUMAN = "require_human"


class DecisionGovernanceMismatchError(ValueError):
    """Raised when governance artifacts do not match bound decision context."""


@dataclass(frozen=True, slots=True)
class DecisionGovernanceDecision:
    """Policy evaluation result for one authoritative decision and action."""

    disposition: DecisionGovernanceDisposition
    decision_ref: AuthoritativeDecisionRef
    action: DecisionExecutionAction
    policy_context: DecisionGovernancePolicyContext
    tenant_id: str

    def __post_init__(self) -> None:
        if type(self.disposition) is not DecisionGovernanceDisposition:
            raise TypeError(
                "DecisionGovernanceDecision.disposition must be DecisionGovernanceDisposition",
            )
        if type(self.decision_ref) is not AuthoritativeDecisionRef:
            raise TypeError(
                "DecisionGovernanceDecision.decision_ref must be AuthoritativeDecisionRef",
            )
        if type(self.action) is not DecisionExecutionAction:
            raise TypeError(
                "DecisionGovernanceDecision.action must be DecisionExecutionAction",
            )
        if type(self.policy_context) is not DecisionGovernancePolicyContext:
            raise TypeError(
                "DecisionGovernanceDecision.policy_context must be DecisionGovernancePolicyContext",
            )
        validate_decision_tenant_id(self.tenant_id)
        if self.decision_ref.identity.tenant_id != self.tenant_id:
            raise ValueError(
                "DecisionGovernanceDecision.tenant_id must match decision_ref identity tenant_id",
            )


@dataclass(frozen=True, slots=True)
class DecisionExecutionAuthorization:
    """Immutable authorization to execute one action for one exact decision version."""

    authorization_id: DecisionExecutionAuthorizationId
    decision_ref: AuthoritativeDecisionRef
    action: DecisionExecutionAction
    policy_context: DecisionGovernancePolicyContext
    tenant_id: str

    def __post_init__(self) -> None:
        validate_decision_execution_authorization_id(self.authorization_id)
        if type(self.decision_ref) is not AuthoritativeDecisionRef:
            raise TypeError(
                "DecisionExecutionAuthorization.decision_ref must be AuthoritativeDecisionRef",
            )
        if type(self.action) is not DecisionExecutionAction:
            raise TypeError(
                "DecisionExecutionAuthorization.action must be DecisionExecutionAction",
            )
        if type(self.policy_context) is not DecisionGovernancePolicyContext:
            raise TypeError(
                "DecisionExecutionAuthorization.policy_context must be DecisionGovernancePolicyContext",
            )
        validate_decision_tenant_id(self.tenant_id)
        if self.decision_ref.identity.tenant_id != self.tenant_id:
            raise ValueError(
                "DecisionExecutionAuthorization.tenant_id must match decision_ref identity tenant_id",
            )


@dataclass(frozen=True, slots=True)
class DecisionGovernanceEvaluationInput(Generic[T]):
    """Canonical input for governance evaluation over one accepted decision."""

    decision: AuthoritativeAcceptedDecision[T]
    action: DecisionExecutionAction
    policy_context: DecisionGovernancePolicyContext

    def __post_init__(self) -> None:
        if type(self.decision) is not AuthoritativeAcceptedDecision:
            raise TypeError(
                "DecisionGovernanceEvaluationInput.decision must be AuthoritativeAcceptedDecision",
            )
        if type(self.action) is not DecisionExecutionAction:
            raise TypeError(
                "DecisionGovernanceEvaluationInput.action must be DecisionExecutionAction",
            )
        if type(self.policy_context) is not DecisionGovernancePolicyContext:
            raise TypeError(
                "DecisionGovernanceEvaluationInput.policy_context must be DecisionGovernancePolicyContext",
            )


def validate_governance_decision_against_input(
    *,
    evaluation_input: DecisionGovernanceEvaluationInput[T],
    governance_decision: DecisionGovernanceDecision,
) -> None:
    """Reject plugin governance output that violates exact identity bindings."""
    if type(evaluation_input) is not DecisionGovernanceEvaluationInput:
        raise TypeError("evaluation_input must be DecisionGovernanceEvaluationInput")
    if type(governance_decision) is not DecisionGovernanceDecision:
        raise TypeError("governance_decision must be DecisionGovernanceDecision")
    expected_ref = authoritative_decision_ref(evaluation_input.decision)
    if not authoritative_decision_refs_match(
        governance_decision.decision_ref,
        expected_ref,
    ):
        raise DecisionGovernanceMismatchError(
            "governance_decision.decision_ref must match evaluation_input decision",
        )
    if governance_decision.action != evaluation_input.action:
        raise DecisionGovernanceMismatchError(
            "governance_decision.action must match evaluation_input action",
        )
    if governance_decision.policy_context != evaluation_input.policy_context:
        raise DecisionGovernanceMismatchError(
            "governance_decision.policy_context must match evaluation_input policy_context",
        )
    if governance_decision.tenant_id != evaluation_input.decision.identity.tenant_id:
        raise DecisionGovernanceMismatchError(
            "governance_decision.tenant_id must match decision identity tenant_id",
        )


def decision_execution_authorization(
    *,
    governance_decision: DecisionGovernanceDecision,
    authorization_id: DecisionExecutionAuthorizationId | None = None,
) -> DecisionExecutionAuthorization:
    """Mint one execution authorization from an ALLOW governance decision."""
    if type(governance_decision) is not DecisionGovernanceDecision:
        raise TypeError("governance_decision must be DecisionGovernanceDecision")
    if governance_decision.disposition is not DecisionGovernanceDisposition.ALLOW:
        raise ValueError(
            "execution authorization requires DecisionGovernanceDisposition.ALLOW",
        )
    resolved_authorization_id = (
        authorization_id
        if authorization_id is not None
        else mint_decision_execution_authorization_id()
    )
    return DecisionExecutionAuthorization(
        authorization_id=resolved_authorization_id,
        decision_ref=governance_decision.decision_ref,
        action=governance_decision.action,
        policy_context=governance_decision.policy_context,
        tenant_id=governance_decision.tenant_id,
    )


def validate_execution_authorization_for_decision(
    *,
    authorization: DecisionExecutionAuthorization,
    decision: AuthoritativeAcceptedDecision[T],
) -> None:
    """Reject stale execution authorization for a different decision version."""
    if type(authorization) is not DecisionExecutionAuthorization:
        raise TypeError("authorization must be DecisionExecutionAuthorization")
    if type(decision) is not AuthoritativeAcceptedDecision:
        raise TypeError("decision must be AuthoritativeAcceptedDecision")
    expected_ref = authoritative_decision_ref(decision)
    if not authoritative_decision_refs_match(
        authorization.decision_ref,
        expected_ref,
    ):
        raise DecisionGovernanceMismatchError(
            "execution authorization decision_ref must match authoritative decision",
        )
    if authorization.tenant_id != decision.identity.tenant_id:
        raise DecisionGovernanceMismatchError(
            "execution authorization tenant_id must match decision identity tenant_id",
        )


def validate_execution_authorization_for_action(
    *,
    authorization: DecisionExecutionAuthorization,
    action: DecisionExecutionAction,
) -> None:
    """Reject execution authorization reuse for a different action."""
    if type(authorization) is not DecisionExecutionAuthorization:
        raise TypeError("authorization must be DecisionExecutionAuthorization")
    if type(action) is not DecisionExecutionAction:
        raise TypeError("action must be DecisionExecutionAction")
    if authorization.action != action:
        raise DecisionGovernanceMismatchError(
            "execution authorization action must match requested action",
        )


def validate_execution_authorization_for_policy_context(
    *,
    authorization: DecisionExecutionAuthorization,
    policy_context: DecisionGovernancePolicyContext,
) -> None:
    """Reject execution authorization reuse under a different policy context."""
    if type(authorization) is not DecisionExecutionAuthorization:
        raise TypeError("authorization must be DecisionExecutionAuthorization")
    if type(policy_context) is not DecisionGovernancePolicyContext:
        raise TypeError("policy_context must be DecisionGovernancePolicyContext")
    if authorization.policy_context != policy_context:
        raise DecisionGovernanceMismatchError(
            "execution authorization policy_context must match requested policy_context",
        )


@runtime_checkable
class DecisionAuthorizationEvaluator(Protocol):
    """Optional substitution seam for custom governance policy evaluation."""

    def evaluate(
        self,
        *,
        evaluation_input: DecisionGovernanceEvaluationInput[T],
    ) -> DecisionGovernanceDecision:
        """Evaluate governance authorization for one accepted decision action."""
        ...


def evaluate_decision_governance_with(
    *,
    evaluator: DecisionAuthorizationEvaluator,
    evaluation_input: DecisionGovernanceEvaluationInput[T],
) -> DecisionGovernanceDecision:
    """Evaluate via a custom evaluator and reject semantically invalid output."""
    if type(evaluation_input) is not DecisionGovernanceEvaluationInput:
        raise TypeError("evaluation_input must be DecisionGovernanceEvaluationInput")
    decision = evaluator.evaluate(evaluation_input=evaluation_input)
    validate_governance_decision_against_input(
        evaluation_input=evaluation_input,
        governance_decision=decision,
    )
    return decision
