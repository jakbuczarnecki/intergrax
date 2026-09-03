# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass

import pytest

from intergrax.contracts.decision_authorization import (
    AuthoritativeDecisionRef,
    DecisionExecutionAction,
    DecisionGovernanceDecision,
    DecisionGovernanceDisposition,
    DecisionGovernanceMismatchError,
    DecisionGovernancePolicyContext,
    authoritative_decision_ref,
    authoritative_decision_refs_match,
    decision_execution_action,
    decision_execution_authorization,
    decision_governance_policy_context,
    mint_decision_execution_authorization_id,
    validate_decision_execution_action_kind,
    validate_decision_execution_authorization_id,
    validate_execution_authorization_for_action,
    validate_execution_authorization_for_decision,
    validate_execution_authorization_for_policy_context,
    validate_governance_decision_against_input,
)
from intergrax.contracts.decision_identity import (
    DecisionExecutionLineage,
    DecisionIdentity,
    DecisionId,
    DecisionScope,
    DecisionVersion,
    initial_decision_version,
    mint_decision_id,
    next_decision_version,
)
from intergrax.contracts.decision_record import (
    AuthoritativeAcceptedDecision,
    CandidateDecision,
    DecisionArtifact,
    DecisionProposalRef,
    candidate_decision_ref,
    decision_lineage_ref,
    decision_proposal_ref,
    decision_version_lineage,
    validate_decision_artifact_kind,
    validate_decision_branch_id,
)
from intergrax.contracts.execution_identity import (
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]


@dataclass(frozen=True, slots=True)
class Payload:
    text: str


def _execution_lineage() -> DecisionExecutionLineage:
    return DecisionExecutionLineage(
        task_id=mint_task_id(),
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
        execution_id=mint_execution_id(),
    )


def _identity(
    *,
    version: DecisionVersion | None = None,
    decision_id: DecisionId | None = None,
    tenant_id: str = "tenant-a",
) -> DecisionIdentity:
    return DecisionIdentity(
        decision_id=mint_decision_id() if decision_id is None else decision_id,
        version=version or initial_decision_version(),
        scope=DecisionScope(namespace="demo", subject="case-1"),
        tenant_id=tenant_id,
        execution=_execution_lineage(),
    )


def _accepted(
    *,
    identity: DecisionIdentity | None = None,
    branch_id: str = "main",
) -> AuthoritativeAcceptedDecision[Payload]:
    resolved_identity = identity or _identity()
    lineage = decision_version_lineage(
        current=decision_lineage_ref(
            resolved_identity.version,
            validate_decision_branch_id(branch_id),
        ),
    )
    return AuthoritativeAcceptedDecision(
        identity=resolved_identity,
        artifact=DecisionArtifact(
            kind=validate_decision_artifact_kind("demo.payload"),
            content=Payload(text="accepted"),
        ),
        lineage=lineage,
    )


def _action(*, kind: str = "tool.side_effect", subject: str = "notify.ops") -> DecisionExecutionAction:
    return decision_execution_action(
        kind=validate_decision_execution_action_kind(kind),
        subject=subject,
    )


def _policy(*, digest: str = "policy-digest-a") -> DecisionGovernancePolicyContext:
    return decision_governance_policy_context(
        policy_provenance_digest=digest,
        matched_rule_ids=("rule.allow.notify",),
    )


def _allow_decision(
    decision: AuthoritativeAcceptedDecision[Payload],
    *,
    action: DecisionExecutionAction | None = None,
    policy: DecisionGovernancePolicyContext | None = None,
) -> DecisionGovernanceDecision:
    return DecisionGovernanceDecision(
        disposition=DecisionGovernanceDisposition.ALLOW,
        decision_ref=authoritative_decision_ref(decision),
        action=action or _action(),
        policy_context=policy or _policy(),
        tenant_id=decision.identity.tenant_id,
    )


def test_valid_allow_mints_authorization() -> None:
    decision = _accepted()
    governance = _allow_decision(decision)
    authorization = decision_execution_authorization(governance_decision=governance)
    validate_execution_authorization_for_decision(
        authorization=authorization,
        decision=decision,
    )
    validate_execution_authorization_for_action(
        authorization=authorization,
        action=governance.action,
    )


def test_deny_does_not_mint_authorization() -> None:
    decision = _accepted()
    governance = DecisionGovernanceDecision(
        disposition=DecisionGovernanceDisposition.DENY,
        decision_ref=authoritative_decision_ref(decision),
        action=_action(),
        policy_context=_policy(),
        tenant_id=decision.identity.tenant_id,
    )
    with pytest.raises(ValueError, match="ALLOW"):
        decision_execution_authorization(governance_decision=governance)


def test_require_human_does_not_mint_authorization() -> None:
    decision = _accepted()
    governance = DecisionGovernanceDecision(
        disposition=DecisionGovernanceDisposition.REQUIRE_HUMAN,
        decision_ref=authoritative_decision_ref(decision),
        action=_action(),
        policy_context=_policy(),
        tenant_id=decision.identity.tenant_id,
    )
    with pytest.raises(ValueError, match="ALLOW"):
        decision_execution_authorization(governance_decision=governance)


def test_wrong_decision_version_rejected() -> None:
    decision_v1 = _accepted()
    governance = _allow_decision(decision_v1)
    authorization = decision_execution_authorization(governance_decision=governance)
    decision_v2 = AuthoritativeAcceptedDecision(
        identity=DecisionIdentity(
            decision_id=decision_v1.identity.decision_id,
            version=next_decision_version(decision_v1.identity.version),
            scope=decision_v1.identity.scope,
            tenant_id=decision_v1.identity.tenant_id,
            execution=decision_v1.identity.execution,
        ),
        artifact=decision_v1.artifact,
        lineage=decision_version_lineage(
            current=decision_lineage_ref(next_decision_version(decision_v1.identity.version)),
            parents=(authoritative_decision_ref(decision_v1).lineage_ref,),
        ),
    )
    with pytest.raises(DecisionGovernanceMismatchError):
        validate_execution_authorization_for_decision(
            authorization=authorization,
            decision=decision_v2,
        )


def test_wrong_action_rejected() -> None:
    decision = _accepted()
    governance = _allow_decision(decision, action=_action(subject="notify.ops"))
    authorization = decision_execution_authorization(governance_decision=governance)
    with pytest.raises(DecisionGovernanceMismatchError):
        validate_execution_authorization_for_action(
            authorization=authorization,
            action=_action(subject="deploy.prod"),
        )


def test_wrong_policy_context_rejected() -> None:
    decision = _accepted()
    governance = _allow_decision(decision, policy=_policy(digest="policy-digest-a"))
    authorization = decision_execution_authorization(governance_decision=governance)
    with pytest.raises(DecisionGovernanceMismatchError):
        validate_execution_authorization_for_policy_context(
            authorization=authorization,
            policy_context=_policy(digest="policy-digest-b"),
        )


def test_wrong_tenant_rejected() -> None:
    decision = _accepted(identity=_identity(tenant_id="tenant-a"))
    governance = _allow_decision(decision)
    authorization = decision_execution_authorization(governance_decision=governance)
    decision_other_tenant = _accepted(identity=_identity(tenant_id="tenant-b"))
    with pytest.raises(DecisionGovernanceMismatchError):
        validate_execution_authorization_for_decision(
            authorization=authorization,
            decision=decision_other_tenant,
        )


def test_wrong_branch_rejected() -> None:
    decision_v1 = _accepted()
    ref_v1 = authoritative_decision_ref(decision_v1)
    identity_v2 = DecisionIdentity(
        decision_id=decision_v1.identity.decision_id,
        version=next_decision_version(decision_v1.identity.version),
        scope=decision_v1.identity.scope,
        tenant_id=decision_v1.identity.tenant_id,
        execution=decision_v1.identity.execution,
    )
    ref_a = decision_proposal_ref(
        identity=identity_v2,
        lineage_ref=decision_lineage_ref(identity_v2.version, validate_decision_branch_id("A")),
    )
    decision_a = AuthoritativeAcceptedDecision(
        identity=identity_v2,
        artifact=decision_v1.artifact,
        lineage=decision_version_lineage(
            current=ref_a.lineage_ref,
            parents=(ref_v1.lineage_ref,),
        ),
    )
    governance = _allow_decision(decision_a)
    authorization = decision_execution_authorization(governance_decision=governance)
    ref_b = decision_proposal_ref(
        identity=identity_v2,
        lineage_ref=decision_lineage_ref(identity_v2.version, validate_decision_branch_id("B")),
    )
    decision_b = AuthoritativeAcceptedDecision(
        identity=identity_v2,
        artifact=decision_v1.artifact,
        lineage=decision_version_lineage(
            current=ref_b.lineage_ref,
            parents=(ref_v1.lineage_ref,),
        ),
    )
    with pytest.raises(DecisionGovernanceMismatchError):
        validate_execution_authorization_for_decision(
            authorization=authorization,
            decision=decision_b,
        )


def test_wrong_execution_lineage_rejected() -> None:
    decision_a = _accepted()
    governance = _allow_decision(decision_a)
    authorization = decision_execution_authorization(governance_decision=governance)
    decision_b = _accepted(
        identity=DecisionIdentity(
            decision_id=decision_a.identity.decision_id,
            version=decision_a.identity.version,
            scope=decision_a.identity.scope,
            tenant_id=decision_a.identity.tenant_id,
            execution=DecisionExecutionLineage(
                task_id=mint_task_id(),
                run_id=mint_run_id(),
                attempt_id=mint_attempt_id(),
                execution_id=mint_execution_id(),
            ),
        ),
    )
    with pytest.raises(DecisionGovernanceMismatchError):
        validate_execution_authorization_for_decision(
            authorization=authorization,
            decision=decision_b,
        )


def test_authorization_id_mint_and_validate() -> None:
    authorization_id = mint_decision_execution_authorization_id()
    assert validate_decision_execution_authorization_id(authorization_id) == authorization_id


def test_authoritative_decision_refs_match_version_and_branch() -> None:
    decision = _accepted()
    ref = authoritative_decision_ref(decision)
    assert authoritative_decision_refs_match(ref, ref)
    assert type(ref) is AuthoritativeDecisionRef
