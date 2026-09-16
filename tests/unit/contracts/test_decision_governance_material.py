# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass

import pytest

from intergrax.contracts.decision_authorization import (
    DecisionGovernanceDecision,
    DecisionGovernanceDisposition,
    authoritative_decision_ref,
    decision_execution_action,
    decision_execution_authorization,
    decision_governance_policy_context,
)
from intergrax.contracts.decision_governance_material import (
    DecisionGovernanceMaterialMismatchError,
    compute_decision_governance_material_digest,
    decision_governance_material_ref_from_accepted,
    validate_decision_governance_material_for_authorization,
    validate_decision_governance_material_for_decision,
)
from intergrax.contracts.decision_identity import (
    DecisionExecutionLineage,
    DecisionIdentity,
    DecisionScope,
    DecisionVersion,
    initial_decision_version,
    mint_decision_id,
    next_decision_version,
)
from intergrax.contracts.decision_record import (
    AuthoritativeAcceptedDecision,
    DecisionArtifact,
    decision_lineage_ref,
    decision_version_lineage,
    validate_decision_artifact_kind,
)
from intergrax.contracts.execution_identity import (
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
)

pytestmark = pytest.mark.unit


@dataclass(frozen=True, slots=True)
class _Payload:
    text: str


def _accepted(*, version: DecisionVersion | None = None) -> AuthoritativeAcceptedDecision[_Payload]:
    resolved_version = version or initial_decision_version()
    lineage = decision_version_lineage(current=decision_lineage_ref(resolved_version))
    identity = DecisionIdentity(
        decision_id=mint_decision_id(),
        version=resolved_version,
        scope=DecisionScope(namespace="test", subject="gr6"),
        tenant_id="tenant-gr6",
        execution=DecisionExecutionLineage(
            task_id=mint_task_id(),
            run_id=mint_run_id(),
            attempt_id=mint_attempt_id(),
            execution_id=mint_execution_id(),
        ),
    )
    return AuthoritativeAcceptedDecision(
        identity=identity,
        artifact=DecisionArtifact(
            kind=validate_decision_artifact_kind("test.payload"),
            content=_Payload(text="ok"),
        ),
        lineage=lineage,
    )


def _action():
    return decision_execution_action(kind="sandbox_row_write", subject="sandbox")


def test_material_digest_stable_for_same_decision() -> None:
    decision = _accepted()
    action = _action()
    left = compute_decision_governance_material_digest(decision=decision, action=action)
    right = compute_decision_governance_material_digest(decision=decision, action=action)
    assert left == right


def test_material_mismatch_on_version() -> None:
    decision_v1 = _accepted()
    action = _action()
    material_v1 = decision_governance_material_ref_from_accepted(
        decision=decision_v1,
        action=action,
    )
    v2 = next_decision_version(decision_v1.identity.version)
    decision_v2 = AuthoritativeAcceptedDecision(
        identity=DecisionIdentity(
            decision_id=decision_v1.identity.decision_id,
            version=v2,
            scope=decision_v1.identity.scope,
            tenant_id=decision_v1.identity.tenant_id,
            execution=decision_v1.identity.execution,
        ),
        artifact=decision_v1.artifact,
        lineage=decision_version_lineage(
            current=decision_lineage_ref(v2),
            parents=(decision_lineage_ref(decision_v1.identity.version),),
        ),
    )
    with pytest.raises(DecisionGovernanceMaterialMismatchError):
        validate_decision_governance_material_for_decision(
            material=material_v1,
            decision=decision_v2,
            action=action,
        )


def test_material_matches_authorization_on_allow() -> None:
    decision = _accepted()
    action = _action()
    policy = decision_governance_policy_context(policy_provenance_digest="policy-gr6")
    governance = DecisionGovernanceDecision(
        disposition=DecisionGovernanceDisposition.ALLOW,
        decision_ref=authoritative_decision_ref(decision),
        action=action,
        policy_context=policy,
        tenant_id=decision.identity.tenant_id,
    )
    authorization = decision_execution_authorization(governance_decision=governance)
    material = decision_governance_material_ref_from_accepted(
        decision=decision,
        action=action,
    )
    validate_decision_governance_material_for_authorization(
        material=material,
        authorization=authorization,
        action=action,
    )


def test_authorization_mismatch_on_wrong_decision_id() -> None:
    decision = _accepted()
    other = _accepted()
    action = _action()
    material = decision_governance_material_ref_from_accepted(
        decision=decision,
        action=action,
    )
    policy = decision_governance_policy_context(policy_provenance_digest="policy-gr6")
    governance = DecisionGovernanceDecision(
        disposition=DecisionGovernanceDisposition.ALLOW,
        decision_ref=authoritative_decision_ref(other),
        action=action,
        policy_context=policy,
        tenant_id=other.identity.tenant_id,
    )
    authorization = decision_execution_authorization(governance_decision=governance)
    with pytest.raises(DecisionGovernanceMaterialMismatchError):
        validate_decision_governance_material_for_authorization(
            material=material,
            authorization=authorization,
            action=action,
        )
