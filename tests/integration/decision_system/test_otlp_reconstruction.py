# © Artur Czarnecki. All rights reserved.

"""DS-E2E-11 — OTLP reconstruction for Decision lifecycle."""

from __future__ import annotations

from pathlib import Path

import pytest

from intergrax.contracts.decision_lifecycle import DecisionLifecycleStage
from intergrax.contracts.decision_revision import decision_revision_policy
from intergrax.runtime.decision_lifecycle_observability import (
    register_decision_lifecycle_domain_signals,
)
from intergrax.runtime.diagnostics.decision_lifecycle_projection import (
    project_decision_lifecycle_snapshot,
)
from intergrax.runtime.execution.active_decision_lifecycle_host import (
    bind_active_decision_lifecycle_host,
    reset_active_decision_lifecycle_host,
)
from testing_support.decision_e2e.environment import (
    docker_daemon_available,
    qualification_strict_required,
)
from tests.integration.runtime.diag_final_otel_support import (
    build_diag_final_product_host,
    build_observability_export_config,
    execute_host_run,
    refresh_collector_output,
    require_docker_for_external_otlp_proof,
    start_collector_process_only,
    stop_collector_process_only,
    wait_for_collector_event_id,
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
from testing_support.decision_e2e.evidence import runtime_event_count_evidence
from testing_support.decision_e2e.verification import build_pass_through_pipeline

pytestmark = [
    pytest.mark.integration,
    pytest.mark.e2e,
    pytest.mark.qualification,
    pytest.mark.external_proof,
    pytest.mark.docker,
    pytest.mark.no_ci,
    pytest.mark.slow,
]


@pytest.mark.asyncio
async def test_ds_e2e_11_otlp_reconstruction(
    decision_e2e_composition,
    decision_e2e_report_collector,
    tmp_path: Path,
) -> None:
    if not docker_daemon_available():
        disposition = QualificationDisposition.BLOCKED
        reason = "docker daemon unavailable for OTLP qualification"
        if qualification_strict_required():
            pytest.fail(reason)
        decision_e2e_report_collector.record(
            DecisionE2EQualificationResult(
                proof_id=DecisionE2EProofId.DS_E2E_11,
                disposition=disposition,
                evidence=(),
                reason=reason,
            ),
        )
        pytest.skip(reason)

    require_docker_for_external_otlp_proof()
    register_decision_lifecycle_domain_signals()
    composition = decision_e2e_composition
    identity = mint_qualification_identity(subject="otlp-reconstruction")
    gate = composition.build_flow_gate(
        pipeline=build_pass_through_pipeline(),
        revision_policy=decision_revision_policy(max_revisions=0),
    )
    collector = start_collector_process_only()
    try:
        token = bind_active_decision_lifecycle_host(composition.lifecycle_host)
        try:
            payload, _ = await run_single_model_producer(
                composition,
                identity=identity,
                task_message="Return recommendation=observe with confidence=high.",
            )
            flow_result = await evaluate_decision_flow(
                composition,
                gate,
                identity=identity,
                payload=payload,
            )
        finally:
            reset_active_decision_lifecycle_host(token)
        events = tuple(composition.event_bus.history)
        snapshot = project_decision_lifecycle_snapshot(events)
        assert snapshot.decision_id == identity.decision_id
        assert snapshot.tenant_id == identity.tenant_id
        assert snapshot.current_stage in {
            DecisionLifecycleStage.FINALIZATION,
            DecisionLifecycleStage.TERMINAL,
        }
        export_config = build_observability_export_config(collector.endpoint)
        host = build_diag_final_product_host(
            tmp_path=tmp_path,
            document_store=__import__(
                "intergrax.integrations._shared.in_memory_document_store",
                fromlist=["InMemoryDocumentStore"],
            ).InMemoryDocumentStore(),
            observability_export=export_config,
            tenant_id=identity.tenant_id,
        )
        run = execute_host_run(
            host,
            tenant_id=identity.tenant_id,
            message="decision-e2e otlp spine",
        )
        event_id = str(run["terminal_event_id"])
        wait_for_collector_event_id(collector, event_id=event_id)
        collector_payload = refresh_collector_output(collector)
        assert collector_payload
        forbidden = (
            identity.scope.subject,
            payload.recommendation,
            "secret",
        )
        for item in forbidden:
            assert item not in collector_payload
    finally:
        stop_collector_process_only(collector)

    decision_e2e_report_collector.record(
        DecisionE2EQualificationResult(
            proof_id=DecisionE2EProofId.DS_E2E_11,
            disposition=QualificationDisposition.PASSED,
            evidence=(runtime_event_count_evidence(len(events)),),
        ),
    )
