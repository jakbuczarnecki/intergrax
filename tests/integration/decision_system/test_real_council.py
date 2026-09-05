# © Artur Czarnecki. All rights reserved.

"""DS-E2E-02 — real multi-model Council."""

from __future__ import annotations

import pytest

from intergrax.contracts.council_strategy import CouncilResolutionDisposition
from intergrax.contracts.decision_lifecycle import DecisionLifecycleStage
from intergrax.contracts.decision_revision import decision_revision_policy
from intergrax.runtime.decision_flow import DecisionFlowHostAction

from testing_support.decision_e2e.composition import (
    evaluate_decision_flow,
    mint_qualification_identity,
)
from testing_support.decision_e2e.council import (
    council_deliberation_input,
    run_council_deliberation,
    run_with_execution_bindings,
    three_participant_strategy,
)
from testing_support.decision_e2e.evidence import decision_identity_evidence
from testing_support.decision_e2e.requirements import qualify_real_multi_model
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
async def test_ds_e2e_02_real_council(
    decision_e2e_composition,
    decision_e2e_report_collector,
) -> None:
    composition = decision_e2e_composition
    identity = mint_qualification_identity(subject="council-qualification")
    deliberation_input = council_deliberation_input(
        identity,
        task_message="Propose recommendation monitor with confidence medium.",
    )
    strategy = three_participant_strategy()

    council_result, invocations = await run_with_execution_bindings(
        composition,
        identity,
        lambda: run_council_deliberation(
            composition,
            strategy=strategy,
            deliberation_input=deliberation_input,
        ),
    )

    assert invocations >= 2
    assert council_result.disposition is CouncilResolutionDisposition.SYNTHESIZED
    assert council_result.candidate is not None
    assert len(council_result.proposal_refs) >= 2

    gate = composition.build_flow_gate(
        pipeline=build_pass_through_pipeline(),
        revision_policy=decision_revision_policy(max_revisions=0),
    )
    flow_result = await run_with_execution_bindings(
        composition,
        identity,
        lambda: evaluate_decision_flow(
            composition,
            gate,
            identity=identity,
            payload=council_result.candidate.artifact.content,
        ),
    )

    assert flow_result.host_action is DecisionFlowHostAction.CONTINUE
    assert flow_result.lifecycle_state.stage is DecisionLifecycleStage.FINALIZATION

    decision_e2e_report_collector.record(
        qualify_real_multi_model(
            council_bindings=(
                composition.environment.producer_evidence,
                composition.environment.council_b_evidence,
                composition.environment.council_c_evidence,
            ),
            evidence=(decision_identity_evidence(identity),),
            reason=composition.environment.independence_level.value,
        ),
    )
