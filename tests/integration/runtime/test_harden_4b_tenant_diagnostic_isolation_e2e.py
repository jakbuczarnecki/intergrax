# © Artur Czarnecki. All rights reserved.

"""HARDEN-4B — product-host multi-tenant diagnostic isolation E2E."""

from __future__ import annotations

from pathlib import Path

import pytest

from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.runtime.diagnostics.diagnostic_read_models import DiagnosticProblemDetail
from intergrax.runtime.observability.operator_wiring import ObservabilityExportOperatorConfig
from tests.integration.runtime.diag_final_otel_support import (
    assert_problem_truth,
    assert_runtime_event_truth,
    build_diag_final_product_host,
    build_read_service,
    execute_host_run,
)

pytestmark = [pytest.mark.integration, pytest.mark.gate]

_TENANT_A = "tenant-a"
_TENANT_B = "tenant-b"


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


def _assert_runtime_event_tenant_isolation(
    composition,
    *,
    run_id: str,
    owning_tenant: str,
    foreign_tenant: str,
) -> None:
    store = composition.runtime.observability.runtime_event_store
    assert store is not None
    assert store.list_for_run(run_id, tenant_id=owning_tenant), (
        f"expected RuntimeEvents for run_id={run_id!r} under tenant={owning_tenant!r}"
    )
    assert store.list_for_run(run_id, tenant_id=foreign_tenant) == [], (
        f"foreign tenant {foreign_tenant!r} must not see run_id={run_id!r}"
    )


def test_harden_4b_same_violation_isolated_between_tenants(
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
        tenant_id=(_TENANT_A, _TENANT_B),
        inject_violation=True,
    )
    read_service = build_read_service(composition)

    run_a = execute_host_run(
        composition,
        tenant_id=_TENANT_A,
        message="harden-4b tenant-a violation",
    )
    run_id_a = str(run_a["run_id"])
    task_id_a = str(run_a["task_id"])
    assert_runtime_event_truth(
        composition,
        tenant_id=_TENANT_A,
        run_id=run_id_a,
        task_id=task_id_a,
    )
    problem_id_a = assert_problem_truth(
        composition,
        tenant_id=_TENANT_A,
        run_id=run_id_a,
    )

    run_b = execute_host_run(
        composition,
        tenant_id=_TENANT_B,
        message="harden-4b tenant-b violation",
    )
    run_id_b = str(run_b["run_id"])
    task_id_b = str(run_b["task_id"])
    assert_runtime_event_truth(
        composition,
        tenant_id=_TENANT_B,
        run_id=run_id_b,
        task_id=task_id_b,
    )
    problem_id_b = assert_problem_truth(
        composition,
        tenant_id=_TENANT_B,
        run_id=run_id_b,
    )

    assert problem_id_a != problem_id_b, "P0 data isolation breach: tenants merged to one Problem"

    list_a = read_service.list_problems(tenant_id=_TENANT_A)
    list_b = read_service.list_problems(tenant_id=_TENANT_B)
    problem_ids_a = {str(problem.problem_id) for problem in list_a.problems}
    problem_ids_b = {str(problem.problem_id) for problem in list_b.problems}
    assert problem_id_a in problem_ids_a
    assert problem_id_b not in problem_ids_a
    assert problem_id_b in problem_ids_b
    assert problem_id_a not in problem_ids_b

    detail_a = read_service.get_problem(tenant_id=_TENANT_A, problem_id=problem_id_a)
    detail_b = read_service.get_problem(tenant_id=_TENANT_B, problem_id=problem_id_b)
    assert detail_a is not None
    assert detail_b is not None

    signature_a = detail_a.grouping_provenance.deterministic_signature
    signature_b = detail_b.grouping_provenance.deterministic_signature
    assert signature_a is not None
    assert signature_b is not None
    assert signature_a == signature_b, "expected same violation class across tenant boundary"

    occurrence_runs_a = _occurrence_run_ids(detail_a)
    occurrence_runs_b = _occurrence_run_ids(detail_b)
    assert run_id_a in occurrence_runs_a
    assert run_id_b not in occurrence_runs_a
    assert run_id_b in occurrence_runs_b
    assert run_id_a not in occurrence_runs_b

    _assert_runtime_event_tenant_isolation(
        composition,
        run_id=run_id_a,
        owning_tenant=_TENANT_A,
        foreign_tenant=_TENANT_B,
    )
    _assert_runtime_event_tenant_isolation(
        composition,
        run_id=run_id_b,
        owning_tenant=_TENANT_B,
        foreign_tenant=_TENANT_A,
    )


def test_harden_4b_cross_tenant_problem_id_read_returns_none(
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
        tenant_id=(_TENANT_A, _TENANT_B),
        inject_violation=True,
    )
    read_service = build_read_service(composition)

    run_a = execute_host_run(
        composition,
        tenant_id=_TENANT_A,
        message="harden-4b tenant-a direct-read negative",
    )
    problem_id_a = assert_problem_truth(
        composition,
        tenant_id=_TENANT_A,
        run_id=str(run_a["run_id"]),
    )

    run_b = execute_host_run(
        composition,
        tenant_id=_TENANT_B,
        message="harden-4b tenant-b direct-read negative",
    )
    problem_id_b = assert_problem_truth(
        composition,
        tenant_id=_TENANT_B,
        run_id=str(run_b["run_id"]),
    )

    assert read_service.get_problem(tenant_id=_TENANT_B, problem_id=problem_id_a) is None
    assert read_service.get_problem(tenant_id=_TENANT_A, problem_id=problem_id_b) is None
