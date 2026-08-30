# © Artur Czarnecki. All rights reserved.

"""HARDEN-3F — canonical HOS external OTLP proof through operator wiring."""

from __future__ import annotations

from pathlib import Path

import pytest

from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.runtime.events.runtime_event import RuntimeEventType
from intergrax.runtime.observability.export_health import ObservabilityExporterHealthStatus
from tests.integration.runtime.diag_final_otel_support import (
    DiagFinalCollectorStack,
    assert_collector_hos_privacy,
    assert_collector_identity_matches_runtime_event,
    assert_collector_unreachable,
    assert_problem_truth,
    assert_runtime_event_truth,
    build_diag_final_product_host,
    build_observability_export_config,
    build_read_service,
    execute_host_run,
    refresh_collector_output,
    require_docker_for_external_otlp_proof,
    start_collector_process_only,
    stop_collector_process_only,
    wait_for_collector_event_id,
    wait_for_exporter_health_snapshot,
    write_proof_artifact,
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.external_proof,
    pytest.mark.no_ci,
]

_TENANT = "diag-final-e2e"
_ARTIFACT_DIR = Path(".tmp/session/diag-final-e2e")


@pytest.fixture
def _stub_host_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    from testing_support.builder import MeteringFakeLLMAdapter

    adapter = MeteringFakeLLMAdapter()

    def _resolve(
        env: object,
        agent_override: object | None = None,
        **_: object,
    ) -> object:
        del env
        return agent_override or adapter

    monkeypatch.setattr(
        "intergrax.applications._shared.llm_resolver.resolve_llm_adapter",
        _resolve,
    )


def test_diag_final_external_otel_spine_proof(
    tmp_path: Path,
    diag_final_collector_stack: DiagFinalCollectorStack,
    _stub_host_llm: None,
) -> None:
    require_docker_for_external_otlp_proof()

    document_store = InMemoryDocumentStore()
    storage_root = tmp_path / "storage"
    storage_root.mkdir(parents=True, exist_ok=True)
    export_config = build_observability_export_config(diag_final_collector_stack.endpoint)

    composition = build_diag_final_product_host(
        tmp_path=storage_root,
        document_store=document_store,
        observability_export=export_config,
        tenant_id=_TENANT,
        inject_violation=True,
    )

    first_run = execute_host_run(
        composition,
        tenant_id=_TENANT,
        message="diag-final collector available",
    )
    run_id = str(first_run["run_id"])
    task_id = str(first_run["task_id"])

    terminal_event = assert_runtime_event_truth(
        composition,
        tenant_id=_TENANT,
        run_id=run_id,
        task_id=task_id,
    )
    problem_id = assert_problem_truth(
        composition,
        tenant_id=_TENANT,
        run_id=run_id,
    )

    collector_text = wait_for_collector_event_id(
        diag_final_collector_stack,
        str(terminal_event.event_id),
    )
    identity_attrs = assert_collector_identity_matches_runtime_event(
        collector_text,
        terminal_event=terminal_event,
    )
    assert_collector_hos_privacy(collector_text)
    assert identity_attrs["intergrax.tenant_id"] == _TENANT
    collector_snapshot_before_outage = collector_text

    baseline_health = wait_for_exporter_health_snapshot(
        composition,
        expected_status=ObservabilityExporterHealthStatus.HEALTHY,
    )
    assert baseline_health.consecutive_failures == 0
    assert baseline_health.recovery_count == 0

    stop_collector_process_only()
    assert_collector_unreachable()

    outage_run = execute_host_run(
        composition,
        tenant_id=_TENANT,
        message="diag-final collector unavailable",
    )
    outage_run_id = str(outage_run["run_id"])
    outage_task_id = str(outage_run["task_id"])
    outage_terminal_event = assert_runtime_event_truth(
        composition,
        tenant_id=_TENANT,
        run_id=outage_run_id,
        task_id=outage_task_id,
    )
    outage_event_id = str(outage_terminal_event.event_id)
    assert_problem_truth(
        composition,
        tenant_id=_TENANT,
        run_id=outage_run_id,
    )

    degraded_health = wait_for_exporter_health_snapshot(
        composition,
        expected_status=ObservabilityExporterHealthStatus.DEGRADED,
        min_consecutive_failures=1,
    )
    assert degraded_health.last_failure_at is not None
    assert degraded_health.last_failure_reason == "exporter_failed"

    collector_text_after_outage = refresh_collector_output(diag_final_collector_stack)
    assert run_id in collector_snapshot_before_outage
    assert run_id in collector_text_after_outage
    assert outage_run_id not in collector_text_after_outage
    assert outage_event_id not in collector_text_after_outage

    start_collector_process_only()

    recovery_run = execute_host_run(
        composition,
        tenant_id=_TENANT,
        message="diag-final collector recovered",
    )
    recovery_run_id = str(recovery_run["run_id"])
    recovery_task_id = str(recovery_run["task_id"])
    recovery_terminal_event = assert_runtime_event_truth(
        composition,
        tenant_id=_TENANT,
        run_id=recovery_run_id,
        task_id=recovery_task_id,
    )
    recovery_event_id = str(recovery_terminal_event.event_id)

    collector_text_after_recovery = wait_for_collector_event_id(
        diag_final_collector_stack,
        recovery_event_id,
    )
    assert_collector_identity_matches_runtime_event(
        collector_text_after_recovery,
        terminal_event=recovery_terminal_event,
    )

    recovered_health = wait_for_exporter_health_snapshot(
        composition,
        expected_status=ObservabilityExporterHealthStatus.HEALTHY,
        min_recovery_count=1,
    )
    assert recovered_health.consecutive_failures == 0
    assert recovered_health.recovery_count >= 1
    assert recovered_health.last_success_at is not None
    assert recovered_health.last_failure_at is not None

    collector_text_final = refresh_collector_output(diag_final_collector_stack)
    assert recovery_event_id in collector_text_final
    assert outage_event_id not in collector_text_final
    assert outage_run_id not in collector_text_final
    assert_collector_hos_privacy(collector_text_final)

    composition.client.close()
    restarted = build_diag_final_product_host(
        tmp_path=storage_root,
        document_store=document_store,
        observability_export=export_config,
        tenant_id=_TENANT,
        inject_violation=False,
    )
    execute_host_run(
        restarted,
        tenant_id=_TENANT,
        message="diag-final restart persistence",
    )

    store = restarted.runtime.observability.runtime_event_store
    assert store is not None
    persisted_first = store.list_for_run(run_id, tenant_id=_TENANT)
    assert any(
        event.event_type is RuntimeEventType.TASK_COMPLETED for event in persisted_first
    )
    assert_problem_truth(
        restarted,
        tenant_id=_TENANT,
        run_id=run_id,
    )
    restarted_read_service = build_read_service(restarted)
    assert restarted_read_service.list_problems(tenant_id=_TENANT).total_count >= 1

    artifact_path = write_proof_artifact(
        _ARTIFACT_DIR,
        run_id=run_id,
        task_id=task_id,
        attempt_id=str(terminal_event.attempt_id),
        execution_id=str(terminal_event.execution_id),
        event_id=str(terminal_event.event_id),
        problem_id=problem_id,
        terminal_event_type=terminal_event.event_type.value,
        collector_received=True,
        collector_excerpt=collector_text_final,
        collector_available=False,
        restart_verified=True,
        identity_verified=True,
        privacy_verified=True,
        real_collector_recovery_verified=True,
        health_degraded_verified=True,
        health_recovered_verified=True,
        outage_event_replay_absent=True,
        recovery_run_id=recovery_run_id,
        recovery_event_id=recovery_event_id,
    )
    assert artifact_path.is_file()
    restarted.client.close()
