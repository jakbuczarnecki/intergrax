# © Artur Czarnecki. All rights reserved.

"""DS-E2E-01 — real single-model Decision path."""

from __future__ import annotations

import pytest

from intergrax.contracts.decision_lifecycle import DecisionLifecycleStage
from intergrax.contracts.decision_revision import decision_revision_policy
from intergrax.contracts.decision_finalization import (
    decision_finalization_key,
    guard_decision_finalization,
    initial_decision_finalize_guard,
)
from intergrax.contracts.decision_record import AuthoritativeAcceptedDecision
from intergrax.runtime.decision_flow import DecisionFlowHostAction
from intergrax.runtime.execution.active_decision_lifecycle_host import (
    bind_active_decision_lifecycle_host,
    reset_active_decision_lifecycle_host,
)
from intergrax.runtime.execution.decision_recovery import persist_terminal_decision_state
from intergrax.contracts.decision_checkpoint import decision_checkpoint_state

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
from testing_support.decision_e2e.evidence import (
    decision_identity_evidence,
    invocation_count_evidence,
    lifecycle_stage_evidence,
    provider_evidence_ref,
    runtime_event_count_evidence,
)
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
async def test_ds_e2e_01_real_single_model_decision(
    decision_e2e_sqlite_composition,
    decision_e2e_report_collector,
) -> None:
    composition = decision_e2e_sqlite_composition
    identity = mint_qualification_identity()
    pipeline = build_pass_through_pipeline()
    gate = composition.build_flow_gate(
        pipeline=pipeline,
        revision_policy=decision_revision_policy(max_revisions=0),
    )
    lifecycle_host, event_bus = composition.lifecycle_for_identity(identity)
    token = bind_active_decision_lifecycle_host(lifecycle_host)
    try:
        payload, invocations = await run_single_model_producer(
            composition,
            identity=identity,
            task_message=(
                "Return recommendation=monitor and confidence=high for a bounded "
                "operational decision."
            ),
        )
        assert invocations >= 1
        flow_result = await evaluate_decision_flow(
            composition,
            gate,
            identity=identity,
            payload=payload,
        )
    finally:
        reset_active_decision_lifecycle_host(token)

    assert flow_result.host_action is DecisionFlowHostAction.CONTINUE
    assert flow_result.accepted_decision is not None
    assert flow_result.lifecycle_state.stage is DecisionLifecycleStage.FINALIZATION

    persistence = composition.persistence
    assert persistence is not None
    accepted = flow_result.accepted_decision
    finalize_key = decision_finalization_key(identity)
    guard = guard_decision_finalization(
        initial_decision_finalize_guard(finalize_key),
        accepted,
    )
    checkpoint = decision_checkpoint_state(
        lifecycle=flow_result.lifecycle_state,
        finalization=guard.state,
    )
    terminal = persist_terminal_decision_state(
        checkpoint_persistence=persistence.checkpoint,
        finalization_persistence=persistence.finalization,
        checkpoint=checkpoint,
    )
    assert terminal.lifecycle.stage is DecisionLifecycleStage.TERMINAL

    result = DecisionE2EQualificationResult(
        proof_id=DecisionE2EProofId.DS_E2E_01,
        disposition=QualificationDisposition.PASSED,
        evidence=(
            provider_evidence_ref(composition.environment.producer_evidence),
            decision_identity_evidence(identity),
            invocation_count_evidence(invocations),
            lifecycle_stage_evidence(terminal.lifecycle.stage),
            runtime_event_count_evidence(len(event_bus.history)),
        ),
    )
    decision_e2e_report_collector.record(result)
