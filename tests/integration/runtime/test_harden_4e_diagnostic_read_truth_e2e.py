# © Artur Czarnecki. All rights reserved.

"""HARDEN-4E — product-host diagnostic read truth: reconstruction, unavailable, fail-closed."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from intergrax.applications._shared.diagnostic_read_wiring import (
    resolve_host_diagnostic_read_dependencies,
)
from intergrax.applications._shared.diagnostic_runtime_wiring import (
    resolve_host_terminal_execution_diagnostic_trigger,
)
from intergrax.contracts.execution_identity import validate_run_id, validate_task_id
from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.runtime.diagnostics.diagnostic_assessment import DiagnosticAssessmentBuilder
from intergrax.runtime.diagnostics.diagnostic_read_models import (
    DiagnosticOccurrenceReadStatus,
    DiagnosticReadUnavailableReason,
)
from intergrax.runtime.diagnostics.execution_reconstruction import ExecutionReconstructor
from intergrax.runtime.diagnostics.lifecycle_analysis import LifecycleAnomalyAnalyzer
from intergrax.runtime.observability.operator_wiring import ObservabilityExportOperatorConfig
from tests.integration.runtime.diag_final_otel_support import (
    assert_problem_truth,
    assert_runtime_event_truth,
    build_diag_final_product_host,
    build_read_service,
    execute_host_run,
)

pytestmark = [pytest.mark.integration, pytest.mark.gate]

_TENANT = "harden-4e-read-truth"
_FOREIGN_TENANT = "harden-4e-foreign"
_OBSERVED_AT = datetime(2026, 8, 30, 12, 0, tzinfo=UTC)


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


def _expected_assessment_for_occurrence(
    composition,
    *,
    tenant_id: str,
    task_id: str,
    run_id: str,
):
    deps = resolve_host_diagnostic_read_dependencies(composition.runtime)
    reconstructor = ExecutionReconstructor(
        runtime_events=deps.runtime_event_persistence,
        causal_evidence=deps.causal_evidence_persistence,
    )
    reconstruction = reconstructor.reconstruct_execution(
        tenant_id,
        validate_task_id(task_id),
        validate_run_id(run_id),
    )
    lifecycle = LifecycleAnomalyAnalyzer().analyze(reconstruction)
    return DiagnosticAssessmentBuilder().assess(reconstruction, lifecycle)


def test_harden_4e_reconstructs_problem_from_canonical_runtime_evidence(
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

    run = execute_host_run(
        composition,
        tenant_id=_TENANT,
        message="harden-4e reconstruction proof",
    )
    run_id = str(run["run_id"])
    task_id = str(run["task_id"])
    terminal = assert_runtime_event_truth(
        composition,
        tenant_id=_TENANT,
        run_id=run_id,
        task_id=task_id,
    )
    attempt_id = str(terminal.attempt_id)
    execution_id = str(terminal.execution_id)

    runtime_store = composition.runtime.observability.runtime_event_store
    assert runtime_store is not None
    assert runtime_store.list_for_run(run_id, tenant_id=_TENANT)

    problem_id = assert_problem_truth(
        composition,
        tenant_id=_TENANT,
        run_id=run_id,
    )

    detail = read_service.get_problem(tenant_id=_TENANT, problem_id=problem_id)
    assert detail is not None
    assert str(detail.problem_id) == problem_id

    matching_occurrences = [
        occurrence
        for occurrence in detail.occurrences
        if occurrence.subject_ref.execution() is not None
        and str(occurrence.subject_ref.execution().run_id) == run_id
    ]
    assert matching_occurrences, f"expected occurrence for run_id={run_id!r}"

    expected = _expected_assessment_for_occurrence(
        composition,
        tenant_id=_TENANT,
        task_id=task_id,
        run_id=run_id,
    )
    for occurrence in matching_occurrences:
        execution = occurrence.subject_ref.execution()
        assert execution is not None
        assert str(execution.run_id) == run_id
        assert str(execution.task_id) == task_id
        assert occurrence.read_status is DiagnosticOccurrenceReadStatus.AVAILABLE
        assert occurrence.unavailable_reason is None
        assert occurrence.assessment is not None
        assert occurrence.assessment == expected
        assert occurrence.assessment.has_findings

    assert attempt_id
    assert execution_id


def test_harden_4e_missing_execution_evidence_returns_unavailable_not_fabricated_diagnosis(
    tmp_path: Path,
    _stub_host_llm: None,
) -> None:
    document_store = InMemoryDocumentStore()
    evidence_root = tmp_path / "evidence-host"
    read_root = tmp_path / "read-host"
    evidence_root.mkdir(parents=True, exist_ok=True)
    read_root.mkdir(parents=True, exist_ok=True)
    export_config = ObservabilityExportOperatorConfig(enabled=False)

    evidence_composition = build_diag_final_product_host(
        tmp_path=evidence_root,
        document_store=document_store,
        observability_export=export_config,
        tenant_id=(_TENANT, _FOREIGN_TENANT),
        inject_violation=True,
    )
    run = execute_host_run(
        evidence_composition,
        tenant_id=_TENANT,
        message="harden-4e missing-evidence proof",
    )
    run_id = str(run["run_id"])
    task_id = str(run["task_id"])
    assert_runtime_event_truth(
        evidence_composition,
        tenant_id=_TENANT,
        run_id=run_id,
        task_id=task_id,
    )
    problem_id = assert_problem_truth(
        evidence_composition,
        tenant_id=_TENANT,
        run_id=run_id,
    )

    evidence_store = evidence_composition.runtime.observability.runtime_event_store
    assert evidence_store is not None
    assert evidence_store.list_for_run(run_id, tenant_id=_TENANT)

    read_composition = build_diag_final_product_host(
        tmp_path=read_root,
        document_store=document_store,
        observability_export=export_config,
        tenant_id=(_TENANT, _FOREIGN_TENANT),
        inject_violation=True,
    )
    read_store = read_composition.runtime.observability.runtime_event_store
    assert read_store is not None
    assert read_store.list_for_run(run_id, tenant_id=_TENANT) == []

    foreign_run = execute_host_run(
        read_composition,
        tenant_id=_FOREIGN_TENANT,
        message="harden-4e foreign tenant decoy events",
    )
    foreign_run_id = str(foreign_run["run_id"])
    assert read_store.list_for_run(foreign_run_id, tenant_id=_FOREIGN_TENANT)

    read_service = build_read_service(read_composition)
    detail = read_service.get_problem(tenant_id=_TENANT, problem_id=problem_id)
    assert detail is not None
    assert str(detail.problem_id) == problem_id

    matching_occurrences = [
        occurrence
        for occurrence in detail.occurrences
        if occurrence.subject_ref.execution() is not None
        and str(occurrence.subject_ref.execution().run_id) == run_id
    ]
    assert matching_occurrences, f"expected persisted occurrence for run_id={run_id!r}"

    for occurrence in matching_occurrences:
        execution = occurrence.subject_ref.execution()
        assert execution is not None
        assert str(execution.run_id) == run_id
        assert str(execution.task_id) == task_id
        assert occurrence.read_status is DiagnosticOccurrenceReadStatus.UNAVAILABLE
        assert occurrence.unavailable_reason is (
            DiagnosticReadUnavailableReason.EXECUTION_EVIDENCE_UNAVAILABLE
        )
        assert occurrence.assessment is None


def test_harden_4e_unsupported_diagnostic_subject_fails_closed_without_problem(
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
        inject_violation=False,
    )
    assert composition.runtime.diagnostic_wiring.attached is True

    run = execute_host_run(
        composition,
        tenant_id=_TENANT,
        message="harden-4e unsupported diagnostic scope proof",
    )
    run_id = str(run["run_id"])
    task_id = str(run["task_id"])
    assert_runtime_event_truth(
        composition,
        tenant_id=_TENANT,
        run_id=run_id,
        task_id=task_id,
    )

    trigger = resolve_host_terminal_execution_diagnostic_trigger(composition.runtime)
    result = trigger.trigger_for_terminal_execution(
        tenant_id=_TENANT,
        task_id=validate_task_id(task_id),
        run_id=validate_run_id(run_id),
        observed_at=_OBSERVED_AT,
    )

    assert len(result.execution_results) == 1
    execution_result = result.execution_results[0]
    assert execution_result.has_runtime_events
    assert not execution_result.assessment.has_findings
    assert result.grouping_result.candidates == ()
    assert result.lifecycle_result.created == ()
    assert result.lifecycle_result.updated == ()

    read_service = build_read_service(composition)
    listed = read_service.list_problems(tenant_id=_TENANT)
    assert listed.total_count == 0
    assert listed.problems == ()
