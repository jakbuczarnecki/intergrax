# © Artur Czarnecki. All rights reserved.

"""DS-E2E-03 — real independent semantic verifier."""

from __future__ import annotations

import pytest

from intergrax.contracts.decision_revision import decision_revision_policy
from intergrax.contracts.decision_verification import VerificationDisposition
from intergrax.runtime.decision_flow import DecisionFlowHostAction
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
    QualificationDisposition,
)
from testing_support.decision_e2e.evidence import (
    decision_identity_evidence,
)
from testing_support.decision_e2e.requirements import qualify_independent_verifier
from testing_support.decision_e2e.payloads import QualificationRecommendation
from testing_support.decision_e2e.verification import (
    build_semantic_verification_pipeline,
    compose_eval_bridge,
)

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
async def test_ds_e2e_03_real_independent_semantic_verifier_pass(
    decision_e2e_composition,
    decision_e2e_report_collector,
) -> None:
    composition = decision_e2e_composition
    identity = mint_qualification_identity(subject="semantic-pass")
    bridge = compose_eval_bridge(composition.tool_wiring)
    pipeline = build_semantic_verification_pipeline(
        tool_bridge=bridge,
        rubric_id="decision_e2e_pass",
        min_score=0.0,
        rubric_criteria=("response includes a clear recommendation",),
        producer_profile_id="profile-producer",
        verifier_profile_id="profile-verifier",
    )
    gate = composition.build_flow_gate(
        pipeline=pipeline,
        revision_policy=decision_revision_policy(max_revisions=0),
    )
    lifecycle_host, _event_bus = composition.lifecycle_for_identity(identity)
    token = bind_active_decision_lifecycle_host(lifecycle_host)
    try:
        payload, _ = await run_single_model_producer(
            composition,
            identity=identity,
            task_message=(
                "Return recommendation=approve with confidence=high and a clear "
                "rationale_summary explaining one explicit bounded operational action."
            ),
        )
        pass_result = await evaluate_decision_flow(
            composition,
            gate,
            identity=identity,
            payload=payload,
        )
    finally:
        reset_active_decision_lifecycle_host(token)

    assert pass_result.verification_result is not None
    assert pass_result.verification_result.disposition is VerificationDisposition.PASSED
    assert pass_result.host_action is DecisionFlowHostAction.CONTINUE

    identity_fail = mint_qualification_identity(subject="semantic-fail")
    fail_gate = composition.build_flow_gate(
        pipeline=build_semantic_verification_pipeline(
            tool_bridge=bridge,
            rubric_id="decision_e2e_fail",
            min_score=0.99,
            producer_profile_id="profile-producer",
            verifier_profile_id="profile-verifier",
        ),
        revision_policy=decision_revision_policy(max_revisions=0),
    )
    fail_lifecycle_host, _ = composition.lifecycle_for_identity(identity_fail)
    token = bind_active_decision_lifecycle_host(fail_lifecycle_host)
    try:
        vague_payload = QualificationRecommendation(
            recommendation="maybe",
            confidence="low",
            rationale_summary="",
        )
        fail_result = await evaluate_decision_flow(
            composition,
            fail_gate,
            identity=identity_fail,
            payload=vague_payload,
        )
    finally:
        reset_active_decision_lifecycle_host(token)

    assert fail_result.verification_result is not None
    assert fail_result.verification_result.disposition is VerificationDisposition.CHALLENGED

    decision_e2e_report_collector.record(
        qualify_independent_verifier(
            producer=composition.environment.producer_evidence,
            verifier=composition.environment.verifier_evidence,
            evidence=(
                decision_identity_evidence(identity),
                decision_identity_evidence(identity_fail),
            ),
            reason=composition.environment.independence_level.value,
        ),
    )
