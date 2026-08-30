# © Artur Czarnecki. All rights reserved.

"""HARDEN-4D — product-host resolve → RESOLVED → recurrence reopen E2E."""

from __future__ import annotations

from pathlib import Path

import pytest

from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.runtime.diagnostics.diagnostic_read_models import DiagnosticProblemDetail
from intergrax.runtime.diagnostics.problem_lifecycle import ProblemStatus
from intergrax.runtime.observability.operator_wiring import ObservabilityExportOperatorConfig
from tests.integration.runtime.diag_final_otel_support import (
    assert_problem_truth,
    assert_runtime_event_truth,
    build_diag_final_product_host,
    build_read_service,
    execute_host_run,
    resolve_problem_via_host_lifecycle,
)

pytestmark = [pytest.mark.integration, pytest.mark.gate]

_TENANT = "harden-4d-lifecycle"


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


def _occurrence_run_ids(detail: DiagnosticProblemDetail) -> set[str]:
    run_ids: set[str] = set()
    for occurrence in detail.occurrences:
        execution = occurrence.subject_ref.execution()
        if execution is not None:
            run_ids.add(str(execution.run_id))
    return run_ids


def _matching_problem_ids(
    read_service,
    *,
    tenant_id: str,
    deterministic_signature: str,
) -> set[str]:
    listed = read_service.list_problems(tenant_id=tenant_id)
    matching: set[str] = set()
    for summary in listed.problems:
        signature = summary.grouping_provenance.deterministic_signature
        if signature is not None and str(signature) == deterministic_signature:
            matching.add(str(summary.problem_id))
    return matching


def test_harden_4d_resolve_then_same_violation_reopens_same_problem(
    tmp_path: Path,
    _stub_host_llm: None,
) -> None:
    document_store = InMemoryDocumentStore()
    storage_root = tmp_path / "storage"
    storage_root.mkdir(parents=True, exist_ok=True)
    export_config = ObservabilityExportOperatorConfig(enabled=False)

    composition = build_diag_final_product_host(
        tmp_path=storage_root,
        document_store=document_store,
        observability_export=export_config,
        tenant_id=_TENANT,
        inject_violation=True,
    )
    assert composition.runtime.diagnostic_wiring.attached is True
    read_service = build_read_service(composition)

    run_1 = execute_host_run(
        composition,
        tenant_id=_TENANT,
        message="harden-4d lifecycle run 1",
    )
    run_id_1 = str(run_1["run_id"])
    task_id_1 = str(run_1["task_id"])
    terminal_1 = assert_runtime_event_truth(
        composition,
        tenant_id=_TENANT,
        run_id=run_id_1,
        task_id=task_id_1,
    )
    execution_id_1 = str(terminal_1.execution_id)

    problem_id = assert_problem_truth(
        composition,
        tenant_id=_TENANT,
        run_id=run_id_1,
    )
    initial = read_service.get_problem(tenant_id=_TENANT, problem_id=problem_id)
    assert initial is not None
    assert initial.status is ProblemStatus.OPEN
    assert initial.occurrence_count >= 1
    first_seen = initial.first_seen_at
    deterministic_signature = initial.grouping_provenance.deterministic_signature
    assert deterministic_signature is not None
    occurrence_count_before = initial.occurrence_count
    version_open = initial.record_version

    resolved = resolve_problem_via_host_lifecycle(
        composition,
        tenant_id=_TENANT,
        problem_id=problem_id,
    )
    assert str(resolved.problem_id) == problem_id
    assert resolved.status is ProblemStatus.RESOLVED

    after_resolve = read_service.get_problem(tenant_id=_TENANT, problem_id=problem_id)
    assert after_resolve is not None
    assert after_resolve.status is ProblemStatus.RESOLVED
    assert after_resolve.first_seen_at == first_seen
    assert after_resolve.occurrence_count == occurrence_count_before
    assert after_resolve.grouping_provenance.deterministic_signature == deterministic_signature
    assert run_id_1 in _occurrence_run_ids(after_resolve)
    version_resolved = after_resolve.record_version
    assert version_resolved > version_open

    resolved_list = read_service.list_problems(
        tenant_id=_TENANT,
        status=ProblemStatus.RESOLVED,
    )
    open_list = read_service.list_problems(
        tenant_id=_TENANT,
        status=ProblemStatus.OPEN,
    )
    resolved_ids = {str(problem.problem_id) for problem in resolved_list.problems}
    open_ids = {str(problem.problem_id) for problem in open_list.problems}
    assert problem_id in resolved_ids
    assert problem_id not in open_ids

    run_2 = execute_host_run(
        composition,
        tenant_id=_TENANT,
        message="harden-4d lifecycle run 2",
    )
    run_id_2 = str(run_2["run_id"])
    task_id_2 = str(run_2["task_id"])
    terminal_2 = assert_runtime_event_truth(
        composition,
        tenant_id=_TENANT,
        run_id=run_id_2,
        task_id=task_id_2,
    )
    execution_id_2 = str(terminal_2.execution_id)
    assert execution_id_2
    assert execution_id_1

    reopened = read_service.get_problem(tenant_id=_TENANT, problem_id=problem_id)
    assert reopened is not None
    assert str(reopened.problem_id) == problem_id
    assert reopened.status is ProblemStatus.OPEN
    assert reopened.occurrence_count > occurrence_count_before
    assert reopened.first_seen_at == first_seen
    assert reopened.last_seen_at >= first_seen
    assert reopened.grouping_provenance.deterministic_signature == deterministic_signature
    assert reopened.record_version > version_resolved

    occurrence_runs = _occurrence_run_ids(reopened)
    assert run_id_1 in occurrence_runs
    assert run_id_2 in occurrence_runs

    matching_ids = _matching_problem_ids(
        read_service,
        tenant_id=_TENANT,
        deterministic_signature=str(deterministic_signature),
    )
    assert matching_ids == {problem_id}
