# © Artur Czarnecki. All rights reserved.

"""HARDEN-4F — real Mongo Problem Store failure + recovery qualification."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from intergrax.runtime.diagnostics.diagnostic_subsystem_failure_evidence import (
    diagnostic_subsystem_failure_observed_for_run,
    is_diagnostic_subsystem_failure_event,
)
from intergrax.runtime.diagnostics.problem_lifecycle import ProblemStatus
from intergrax.runtime.events.runtime_event import RuntimeEventType
from tests.integration.applications.architecture.harden_4f_mongo_support import (
    Harden4FMongoHostComposition,
    assert_host_uses_document_store_problem_persistence,
    build_harden_4f_mongo_product_host,
    build_read_service,
    cleanup_proof_tenant,
    create_proof_document_store,
    ensure_mongo_running,
    execute_mongo_host_run,
    occurrence_run_ids,
    proof_env,
    read_problem_via_fresh_store_persistence,
    require_docker_for_harden_4f_proof,
    start_mongo_container,
    stop_mongo_container,
    wait_until_mongo_reachable,
    wait_until_mongo_unreachable,
)
from tests.integration.runtime.diag_final_otel_support import assert_runtime_event_truth

pytestmark = [
    pytest.mark.integration,
    pytest.mark.external_proof,
    pytest.mark.network,
    pytest.mark.no_ci,
]

_TENANT = "harden-4f-mongo-tenant"
_HOST_PROCESS_ID: int | None = None


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


@pytest.fixture
def mongo_proof_environment(monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    env = proof_env()
    for key, value in env.items():
        if key.startswith("INTERGRAX_MONGODB_") or key == "PYTHONPATH":
            monkeypatch.setenv(key, value)
    return env


def _assert_violation_runtime_event(
    composition: Harden4FMongoHostComposition,
    *,
    tenant_id: str,
    run_id: str,
) -> None:
    store = composition.runtime.observability.runtime_event_store
    assert store is not None
    events = store.list_for_run(run_id, tenant_id=tenant_id)
    violation_events = [
        event for event in events if event.event_type is RuntimeEventType.RETRY_SCHEDULED
    ]
    assert violation_events, "expected deterministic RETRY_SCHEDULED violation RuntimeEvent"


def _assert_problem_occurrence_for_run(
    read_service,
    *,
    tenant_id: str,
    run_id: str,
) -> str:
    listed = read_service.list_problems(tenant_id=tenant_id)
    for summary in listed.problems:
        detail = read_service.get_problem(
            tenant_id=tenant_id,
            problem_id=summary.problem_id,
        )
        assert detail is not None
        occurrence_runs: set[str] = set()
        for occurrence in detail.occurrences:
            execution = occurrence.subject_ref.execution()
            if execution is not None:
                occurrence_runs.add(str(execution.run_id))
        if run_id in occurrence_runs:
            return str(summary.problem_id)
    raise AssertionError(f"no central Problem occurrence for run_id={run_id!r}")


def _assert_diagnostic_subsystem_failure_evidence(
    composition: Harden4FMongoHostComposition,
    *,
    tenant_id: str,
    run_id: str,
    task_id: str,
    execution_id: str,
) -> None:
    store = composition.runtime.observability.runtime_event_store
    assert store is not None
    assert diagnostic_subsystem_failure_observed_for_run(
        store,
        tenant_id=tenant_id,
        run_id=run_id,
    )
    failure_events = [
        event
        for event in store.list_for_run(run_id, tenant_id=tenant_id)
        if is_diagnostic_subsystem_failure_event(event)
    ]
    assert len(failure_events) == 1
    failure = failure_events[0]
    assert failure.tenant_id == tenant_id
    assert str(failure.task_id) == task_id
    assert str(failure.run_id) == run_id
    assert str(failure.execution_id) == execution_id
    assert failure.payload.get("error_type")


def test_harden_4f_mongo_problem_store_failure_and_recovery_e2e(
    tmp_path: Path,
    mongo_proof_environment: dict[str, str],
    _stub_host_llm: None,
) -> None:
    """
    HARDEN-4F external proof:

    PHASE 1 — Mongo UP: HTTP execution → canonical RuntimeEvents → Mongo Problem write/read
    PHASE 2 — Mongo DOWN: business survives, evidence survives, Problem write fails visibly
    PHASE 3 — Mongo UP: same host recovers, new write succeeds, no outage replay
    """
    del mongo_proof_environment
    require_docker_for_harden_4f_proof()
    ensure_mongo_running()

    global _HOST_PROCESS_ID
    document_store = create_proof_document_store()
    storage_root = tmp_path / "storage"
    storage_root.mkdir(parents=True, exist_ok=True)
    composition = build_harden_4f_mongo_product_host(
        tmp_path=storage_root,
        document_store=document_store,
        tenant_id=_TENANT,
        inject_violation=True,
    )
    _HOST_PROCESS_ID = os.getpid()
    assert_host_uses_document_store_problem_persistence(composition)
    read_service = build_read_service(composition)

    # PHASE 1 — baseline Mongo UP
    baseline_run = execute_mongo_host_run(
        composition,
        tenant_id=_TENANT,
        message="harden-4f baseline mongo up",
    )
    run_id_baseline = str(baseline_run["run_id"])
    task_id_baseline = str(baseline_run["task_id"])
    terminal_baseline = assert_runtime_event_truth(
        composition,  # type: ignore[arg-type]
        tenant_id=_TENANT,
        run_id=run_id_baseline,
        task_id=task_id_baseline,
    )
    execution_id_baseline = str(terminal_baseline.execution_id)
    _assert_violation_runtime_event(
        composition,
        tenant_id=_TENANT,
        run_id=run_id_baseline,
    )

    problem_id_baseline = _assert_problem_occurrence_for_run(
        read_service,
        tenant_id=_TENANT,
        run_id=run_id_baseline,
    )
    baseline_detail = read_service.get_problem(tenant_id=_TENANT, problem_id=problem_id_baseline)
    assert baseline_detail is not None
    deterministic_signature = baseline_detail.grouping_provenance.deterministic_signature
    assert deterministic_signature is not None
    provider_baseline = read_problem_via_fresh_store_persistence(
        tenant_id=_TENANT,
        problem_id=problem_id_baseline,
    )
    assert provider_baseline is not None
    assert provider_baseline.status is ProblemStatus.OPEN
    assert provider_baseline.occurrence_count >= 1
    assert run_id_baseline in occurrence_run_ids(provider_baseline)
    baseline_occurrence_count = provider_baseline.occurrence_count

    # PHASE 2 — Mongo DOWN
    stop_mongo_container()
    wait_until_mongo_unreachable()

    outage_run = execute_mongo_host_run(
        composition,
        tenant_id=_TENANT,
        message="harden-4f mongo outage",
    )
    run_id_outage = str(outage_run["run_id"])
    task_id_outage = str(outage_run["task_id"])
    terminal_outage = assert_runtime_event_truth(
        composition,  # type: ignore[arg-type]
        tenant_id=_TENANT,
        run_id=run_id_outage,
        task_id=task_id_outage,
    )
    execution_id_outage = str(terminal_outage.execution_id)
    _assert_violation_runtime_event(
        composition,
        tenant_id=_TENANT,
        run_id=run_id_outage,
    )
    _assert_diagnostic_subsystem_failure_evidence(
        composition,
        tenant_id=_TENANT,
        run_id=run_id_outage,
        task_id=task_id_outage,
        execution_id=execution_id_outage,
    )

    # PHASE 3 — Mongo UP recovery on same host/process
    start_mongo_container()
    wait_until_mongo_reachable()
    assert os.getpid() == _HOST_PROCESS_ID

    recovery_run = execute_mongo_host_run(
        composition,
        tenant_id=_TENANT,
        message="harden-4f mongo recovery",
    )
    run_id_recovery = str(recovery_run["run_id"])
    task_id_recovery = str(recovery_run["task_id"])
    terminal_recovery = assert_runtime_event_truth(
        composition,  # type: ignore[arg-type]
        tenant_id=_TENANT,
        run_id=run_id_recovery,
        task_id=task_id_recovery,
    )
    execution_id_recovery = str(terminal_recovery.execution_id)
    _assert_violation_runtime_event(
        composition,
        tenant_id=_TENANT,
        run_id=run_id_recovery,
    )

    problem_id_recovery = _assert_problem_occurrence_for_run(
        read_service,
        tenant_id=_TENANT,
        run_id=run_id_recovery,
    )
    assert problem_id_recovery == problem_id_baseline

    provider_after_recovery = read_problem_via_fresh_store_persistence(
        tenant_id=_TENANT,
        problem_id=problem_id_baseline,
    )
    assert provider_after_recovery is not None
    assert provider_after_recovery.status is ProblemStatus.OPEN
    assert provider_after_recovery.occurrence_count == baseline_occurrence_count + 1
    recovered_runs = occurrence_run_ids(provider_after_recovery)
    assert run_id_baseline in recovered_runs
    assert run_id_recovery in recovered_runs
    assert run_id_outage not in recovered_runs

    detail = read_service.get_problem(tenant_id=_TENANT, problem_id=problem_id_baseline)
    assert detail is not None
    assert str(detail.problem_id) == problem_id_baseline
    assert detail.grouping_provenance.deterministic_signature == deterministic_signature

    document_store.close()
    cleanup_proof_tenant(tenant_id=_TENANT)

    # Replay semantics: outage occurrence NOT_REPLAYED (no automatic replay contract).
    assert execution_id_baseline
    assert execution_id_outage
    assert execution_id_recovery
