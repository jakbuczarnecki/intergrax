# © Artur Czarnecki. All rights reserved.

"""DS-E2E-05 — governed ALLOW/DENY side effect."""

from __future__ import annotations

import pytest

from intergrax.contracts.decision_authorization import (
    decision_execution_action,
    decision_governance_policy_context,
)
from intergrax.contracts.decision_identity import next_decision_version
from intergrax.contracts.decision_revision import decision_revision_policy
from intergrax.runtime.decision_flow import DecisionFlowGovernanceSpec, DecisionFlowHostAction
from intergrax.runtime.execution.active_decision_lifecycle_host import (
    bind_active_decision_lifecycle_host,
    reset_active_decision_lifecycle_host,
)

from testing_support.decision_e2e.composition import (
    evaluate_decision_flow,
    mint_qualification_identity,
    run_single_model_producer,
)
from testing_support.decision_e2e.contracts import (
    DecisionE2EProofId,
    DecisionE2EQualificationResult,
    QualificationDisposition,
)
from testing_support.decision_e2e.evidence import decision_identity_evidence
from testing_support.decision_e2e.governance import PolicyGovernanceEvaluator, SandboxSideEffectStore
from testing_support.decision_e2e.verification import build_pass_through_pipeline

pytestmark = [
    pytest.mark.integration,
    pytest.mark.e2e,
    pytest.mark.qualification,
    pytest.mark.external_provider,
    pytest.mark.network,
    pytest.mark.no_ci,
    pytest.mark.slow,
]


@pytest.mark.asyncio
async def test_ds_e2e_05_governed_side_effect(
    decision_e2e_composition,
    decision_e2e_report_collector,
) -> None:
    composition = decision_e2e_composition
    store = SandboxSideEffectStore()
    action = decision_execution_action(kind="sandbox_row_write")
    policy = decision_governance_policy_context(policy_id="decision_e2e_sandbox")
    allow_gate = composition.build_flow_gate(
        pipeline=build_pass_through_pipeline(),
        revision_policy=decision_revision_policy(max_revisions=0),
        governance_spec=DecisionFlowGovernanceSpec(
            action=action,
            policy_context=policy,
            evaluator=PolicyGovernanceEvaluator(
                action=action,
                policy_context=policy,
                allow=True,
            ),
        ),
    )
    deny_gate = composition.build_flow_gate(
        pipeline=build_pass_through_pipeline(),
        revision_policy=decision_revision_policy(max_revisions=0),
        governance_spec=DecisionFlowGovernanceSpec(
            action=action,
            policy_context=policy,
            evaluator=PolicyGovernanceEvaluator(
                action=action,
                policy_context=policy,
                allow=False,
            ),
        ),
    )
    identity = mint_qualification_identity(subject="governance-allow")
    token = bind_active_decision_lifecycle_host(composition.lifecycle_host)
    try:
        payload, _ = await run_single_model_producer(
            composition,
            identity=identity,
            task_message="Return recommendation=allow with confidence=high.",
        )
        allow_result = await evaluate_decision_flow(
            composition,
            allow_gate,
            identity=identity,
            payload=payload,
        )
    finally:
        reset_active_decision_lifecycle_host(token)

    assert allow_result.host_action is DecisionFlowHostAction.CONTINUE
    assert allow_result.accepted_decision is not None
    store.execute_allow(
        decision=allow_result.accepted_decision,
        action_kind=str(action.kind),
    )
    assert (
        store.count_for_decision_version(
            tenant_id=identity.tenant_id,
            decision_id=str(identity.decision_id),
            decision_version=identity.version.value,
        )
        == 1
    )

    deny_identity = mint_qualification_identity(subject="governance-deny")
    token = bind_active_decision_lifecycle_host(composition.lifecycle_host)
    try:
        deny_payload, _ = await run_single_model_producer(
            composition,
            identity=deny_identity,
            task_message="Return recommendation=deny with confidence=high.",
        )
        deny_result = await evaluate_decision_flow(
            composition,
            deny_gate,
            identity=deny_identity,
            payload=deny_payload,
        )
    finally:
        reset_active_decision_lifecycle_host(token)
    assert deny_result.host_action is DecisionFlowHostAction.BLOCK
    assert (
        store.count_for_decision_version(
            tenant_id=deny_identity.tenant_id,
            decision_id=str(deny_identity.decision_id),
            decision_version=deny_identity.version.value,
        )
        == 0
    )

    stale_version = next_decision_version(identity.version)
    assert (
        store.count_for_decision_version(
            tenant_id=identity.tenant_id,
            decision_id=str(identity.decision_id),
            decision_version=stale_version.value,
        )
        == 0
    )

    decision_e2e_report_collector.record(
        DecisionE2EQualificationResult(
            proof_id=DecisionE2EProofId.DS_E2E_05,
            disposition=QualificationDisposition.PASSED,
            evidence=(decision_identity_evidence(identity),),
        ),
    )
