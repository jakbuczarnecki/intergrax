# © Artur Czarnecki. All rights reserved.

"""R2-H2-Q1 embedded harness elimination and receipt-based gate regression tests."""

from __future__ import annotations

import ast
from pathlib import Path

from testing_support.execution_qualification.catalog.composition import (
    build_default_qualification_catalog,
)
from testing_support.execution_qualification.catalog.mandatory_sources import (
    NPSC5E_R2_H2_Q1_ORCHESTRATOR_PATH,
)
from testing_support.execution_qualification.catalog.profile_builders import (
    NPSC5E_R2_PROFILE_ID,
    NPSC5F_FINAL_PROFILE_ID,
)
from testing_support.execution_qualification.contracts import (
    QualificationRunConfig,
    QualificationRunStatus,
    QualificationSuiteStatus,
)
from testing_support.execution_qualification.coordinator import QualificationCoordinator
from testing_support.execution_qualification.embedded_harness_guard import (
    inventory_embedded_harness_in_module,
)
from testing_support.execution_qualification.embedded_harness_kexpr import (
    R2_H2_Q1_EMBEDDED_HARNESS_KEXPR,
    embedded_harness_test_names,
)
from testing_support.execution_qualification.evidence_provider import (
    PlanRunQualificationEvidenceProvider,
    evaluate_final_gate_from_evidence,
)
from testing_support.execution_qualification.final_semantic_pytest import (
    NPSC5E_R2_H2_Q1_SEMANTIC_SUITE_ID,
    npsc5e_r2_h2_q1_embedded_predecessor_suite_ids,
    npsc5e_r2_h2_q1_semantic_pytest_arguments,
    pytest_arguments_exclude_embedded_harness,
)
from testing_support.execution_qualification.plan_runner import (
    run_qualification_execution_plan,
)

from .fake_executor import FakeQualificationSuiteExecutor

_R2_H2_Q1_MODULE = Path(NPSC5E_R2_H2_Q1_ORCHESTRATOR_PATH)


def test_r2_h2_q1_module_inventory_embedded_harness(repo_root: Path) -> None:
    entries = inventory_embedded_harness_in_module(repo_root / _R2_H2_Q1_MODULE)
    assert any(e.test_function == "test_mandatory_frozen_suite_passes" for e in entries)


def test_canonical_r2_h2_q1_leaf_excludes_embedded_harness() -> None:
    catalog = build_default_qualification_catalog()
    compiled = catalog.compile_profile(NPSC5E_R2_PROFILE_ID)
    suite = compiled.suite_by_id[NPSC5E_R2_H2_Q1_SEMANTIC_SUITE_ID]
    assert suite.pytest_arguments == npsc5e_r2_h2_q1_semantic_pytest_arguments()
    assert pytest_arguments_exclude_embedded_harness(suite.pytest_arguments)
    assert R2_H2_Q1_EMBEDDED_HARNESS_KEXPR in suite.pytest_arguments


def test_r2_h2_q1_gate_requires_embedded_predecessor_receipts() -> None:
    catalog = build_default_qualification_catalog()
    compiled = catalog.compile_profile(NPSC5E_R2_PROFILE_ID)
    gate_id = f"npsc5e-r2.requires.{NPSC5E_R2_H2_Q1_SEMANTIC_SUITE_ID}"
    gate = next(g for g in compiled.graph.gates if g.gate_id == gate_id)
    expected = (
        NPSC5E_R2_H2_Q1_SEMANTIC_SUITE_ID,
        *npsc5e_r2_h2_q1_embedded_predecessor_suite_ids(),
    )
    assert gate.requires == expected


def test_r2_h2_q1_predecessor_fail_propagation_fail_closed(
    repo_root: Path,
    run_artifact_root: Path,
) -> None:
    catalog = build_default_qualification_catalog()
    compiled = catalog.compile_profile(NPSC5E_R2_PROFILE_ID)
    fake = FakeQualificationSuiteExecutor({})
    fake.set_suite_status("npsc5e-r3.mandatory.p0a", QualificationSuiteStatus.FAIL)
    config = QualificationRunConfig(
        repo_root=repo_root,
        max_parallel=2,
        run_artifact_root=run_artifact_root,
        suite_timeout_seconds=30.0,
        run_id="r2-h2-q1-gate-fail",
    )
    result = run_qualification_execution_plan(
        compiled.plan,
        config,
        coordinator=QualificationCoordinator(executor=fake),
    )
    assert result.status is QualificationRunStatus.FAIL


def test_r2_h2_q1_semantic_skip_propagation_fail_closed(
    repo_root: Path,
    run_artifact_root: Path,
) -> None:
    catalog = build_default_qualification_catalog()
    compiled = catalog.compile_profile(NPSC5E_R2_PROFILE_ID)
    fake = FakeQualificationSuiteExecutor({})
    fake.set_suite_status(
        NPSC5E_R2_H2_Q1_SEMANTIC_SUITE_ID,
        QualificationSuiteStatus.SKIP,
    )
    config = QualificationRunConfig(
        repo_root=repo_root,
        max_parallel=2,
        run_artifact_root=run_artifact_root,
        suite_timeout_seconds=30.0,
        run_id="r2-h2-q1-skip",
    )
    result = run_qualification_execution_plan(
        compiled.plan,
        config,
        coordinator=QualificationCoordinator(executor=fake),
    )
    assert result.status is QualificationRunStatus.FAIL


def test_r2_h2_q1_evidence_gate_all_predecessors_pass(
    repo_root: Path,
    run_artifact_root: Path,
) -> None:
    catalog = build_default_qualification_catalog()
    compiled = catalog.compile_profile(NPSC5E_R2_PROFILE_ID)
    fake = FakeQualificationSuiteExecutor({})
    config = QualificationRunConfig(
        repo_root=repo_root,
        max_parallel=2,
        run_artifact_root=run_artifact_root,
        suite_timeout_seconds=30.0,
        run_id="r2-h2-q1-evidence",
    )
    result = run_qualification_execution_plan(
        compiled.plan,
        config,
        coordinator=QualificationCoordinator(executor=fake),
    )
    provider = PlanRunQualificationEvidenceProvider.from_plan_run(
        result,
        run_id=config.run_id,
    )
    gate = evaluate_final_gate_from_evidence(
        provider=provider,
        dependency_ids=(
            NPSC5E_R2_H2_Q1_SEMANTIC_SUITE_ID,
            *npsc5e_r2_h2_q1_embedded_predecessor_suite_ids(),
        ),
    )
    assert gate.status is QualificationSuiteStatus.PASS


def test_npsc5f_final_no_duplicate_r2_h2_q1_predecessor_execution(
    repo_root: Path,
    run_artifact_root: Path,
) -> None:
    catalog = build_default_qualification_catalog()
    compiled = catalog.compile_profile(NPSC5F_FINAL_PROFILE_ID)
    fake = FakeQualificationSuiteExecutor({})
    config = QualificationRunConfig(
        repo_root=repo_root,
        max_parallel=2,
        run_artifact_root=run_artifact_root,
        suite_timeout_seconds=30.0,
        run_id="r2-h2-q1-dedup",
    )
    run_qualification_execution_plan(
        compiled.plan,
        config,
        coordinator=QualificationCoordinator(executor=fake),
    )
    for suite_id in compiled.plan.leaf_suite_ids:
        assert fake.invocation_counts.get(suite_id, 0) == 1


def test_r2_h2_q1_semantic_tests_not_in_harness_exclusion(repo_root: Path) -> None:
    source = (repo_root / _R2_H2_Q1_MODULE).read_text(encoding="utf-8")
    tree = ast.parse(source)
    names = {
        node.name
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_")
    }
    harness = embedded_harness_test_names()
    semantic = names - harness
    assert "test_canonical_predecessor_shas_recorded" in semantic
    assert "test_persistence_contract_exposes_revision_cas" in semantic
    assert semantic
