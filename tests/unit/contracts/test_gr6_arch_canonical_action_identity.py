# © Artur Czarnecki. All rights reserved.

"""GR-6-ARCH — canonical action identity across Decision, Governance, External Work."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from external_contractor_adapter.side_effect_actions import (
    ACTION_ACCEPT_QUOTE,
    ACTION_CANCEL_EXTERNAL_WORK,
    ACTION_CREATE_EXTERNAL_WORK,
)
from intergrax.contracts.decision_authorization import (
    decision_execution_action,
    validate_decision_execution_action_kind,
)
from intergrax.contracts.decision_identity import (
    DecisionExecutionLineage,
    DecisionIdentity,
    DecisionScope,
    initial_decision_version,
    mint_decision_id,
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
from intergrax.contracts.decision_governance_material import (
    decision_governance_material_ref_from_accepted,
)
from intergrax.contracts.decision_requirement_policy import DecisionRequirementContext
from intergrax.contracts.meaningful_side_effect import (
    MeaningfulSideEffectKind,
    MeaningfulSideEffectRequest,
)
from intergrax.runtime.decision_governance_material import (
    assert_decision_governance_material_bound,
)
from intergrax.runtime.governance.decision_requirement_policy import (
    decision_governed_side_effect_requirement_policy,
)
from intergrax.runtime.governance.decision_requirement_policy import (
    classify_decision_requirement,
)
from intergrax.contracts.canonical_inner_governance import CanonicalInnerGovernanceViolation

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_TENANT = "tenant-gr6-arch"
_TASK = mint_task_id()
_RUN = mint_run_id()
_ATTEMPT = mint_attempt_id()
_EXEC = mint_execution_id()
_SCOPE = "external_work.mutate"


@dataclass(frozen=True, slots=True)
class _Payload:
    text: str


def _accepted_decision() -> AuthoritativeAcceptedDecision[_Payload]:
    version = initial_decision_version()
    lineage = decision_version_lineage(current=decision_lineage_ref(version))
    identity = DecisionIdentity(
        decision_id=mint_decision_id(),
        version=version,
        scope=DecisionScope(namespace="test", subject="gr6-arch"),
        tenant_id=_TENANT,
        execution=DecisionExecutionLineage(
            task_id=_TASK,
            run_id=_RUN,
            attempt_id=_ATTEMPT,
            execution_id=_EXEC,
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


def _side_effect(action: str) -> MeaningfulSideEffectRequest:
    return MeaningfulSideEffectRequest(
        action=action,
        kinds=(MeaningfulSideEffectKind.COMMITMENT,),
        side_effect_scope_id=_SCOPE,
        task_id=_TASK,
        run_id=_RUN,
        attempt_id=_ATTEMPT,
        execution_id=_EXEC,
        tenant_id=_TENANT,
    )


def test_external_work_actions_are_valid_decision_execution_action_kinds() -> None:
    for kind in (
        ACTION_ACCEPT_QUOTE,
        ACTION_CREATE_EXTERNAL_WORK,
        ACTION_CANCEL_EXTERNAL_WORK,
    ):
        assert validate_decision_execution_action_kind(kind) == kind


def test_decision_material_bound_action_matches_side_effect_action() -> None:
    action = decision_execution_action(
        kind=ACTION_ACCEPT_QUOTE,
        subject="quote-1",
    )
    decision = _accepted_decision()
    material = decision_governance_material_ref_from_accepted(
        decision=decision,
        action=action,
    )
    bound = _side_effect(ACTION_ACCEPT_QUOTE).model_copy(
        update={"decision_governance_material": material},
    )
    assert_decision_governance_material_bound(bound)


def test_mismatched_side_effect_action_fails_closed() -> None:
    action = decision_execution_action(
        kind=ACTION_ACCEPT_QUOTE,
        subject="quote-1",
    )
    material = decision_governance_material_ref_from_accepted(
        decision=_accepted_decision(),
        action=action,
    )
    wrong = _side_effect(ACTION_CREATE_EXTERNAL_WORK).model_copy(
        update={"decision_governance_material": material},
    )
    with pytest.raises(CanonicalInnerGovernanceViolation):
        assert_decision_governance_material_bound(wrong)


def test_legacy_screaming_snake_action_kind_is_rejected() -> None:
    with pytest.raises(ValueError, match="DecisionExecutionActionKind"):
        validate_decision_execution_action_kind("ACCEPT_QUOTE")


def test_decision_requirement_policy_distinguishes_accept_from_create_and_cancel() -> None:
    policy = decision_governed_side_effect_requirement_policy(
        required_actions=frozenset({ACTION_ACCEPT_QUOTE}),
    )
    accept_ctx = DecisionRequirementContext(
        operation_id="submit_quote_acceptance",
        action=ACTION_ACCEPT_QUOTE,
        kinds=(MeaningfulSideEffectKind.COMMITMENT,),
        side_effect_scope_id=_SCOPE,
        task_id=_TASK,
        run_id=_RUN,
        attempt_id=_ATTEMPT,
        execution_id=_EXEC,
    )
    create_ctx = accept_ctx.model_copy(update={"action": ACTION_CREATE_EXTERNAL_WORK})
    cancel_ctx = accept_ctx.model_copy(update={"action": ACTION_CANCEL_EXTERNAL_WORK})
    from intergrax.contracts.decision_requirement_policy import DecisionRequirement

    assert classify_decision_requirement(policy, accept_ctx) is DecisionRequirement.REQUIRED
    assert classify_decision_requirement(policy, create_ctx) is DecisionRequirement.NOT_REQUIRED
    assert classify_decision_requirement(policy, cancel_ctx) is DecisionRequirement.NOT_REQUIRED


def test_alternate_decision_requirement_policy_is_pluggable() -> None:
    from intergrax.contracts.decision_requirement_policy import (
        DecisionRequirement,
        DecisionRequirementPolicy,
    )

    class _RequireCreateOnly:
        def evaluate(self, context: DecisionRequirementContext) -> DecisionRequirement:
            if context.action == ACTION_CREATE_EXTERNAL_WORK:
                return DecisionRequirement.REQUIRED
            return DecisionRequirement.NOT_REQUIRED

    policy: DecisionRequirementPolicy = _RequireCreateOnly()
    ctx = DecisionRequirementContext(
        operation_id="create_work",
        action=ACTION_CREATE_EXTERNAL_WORK,
        kinds=(MeaningfulSideEffectKind.MUTATION,),
        side_effect_scope_id=_SCOPE,
        task_id=_TASK,
        run_id=_RUN,
        attempt_id=_ATTEMPT,
        execution_id=_EXEC,
    )
    assert classify_decision_requirement(policy, ctx) is DecisionRequirement.REQUIRED
